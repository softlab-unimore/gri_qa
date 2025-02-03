import torch
import os
import functools
import logging
import time

from datetime import datetime
from typing import List, Union, Any
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

from connectors import PgVectorConnector

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.vector_store")

class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct", max_seq_length: int = 8192):
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.max_seq_length = max_seq_length

    def embed_documents(self, texts: list) -> list:
        # Encode documents using model.encode
        return self.model.encode(texts)

    def embed_query(self, query: str) -> list:
        # Encode query using model.encode
        return self.model.encode([query])[0]  # Returns the embedding for a single query

class Handler(object):
    def __init__(self, args):
        super(object, self).__init__()

    def hash_doc(self, doc):
        content = doc.page_content
        metadata = doc.metadata

        input_hash = content + ''.join(list(map(lambda x: str(x), list(metadata.values()))))
        return str(hash(input_hash))


class VectorStoreHandler(Handler):
    def __init__(self, args):
        super(VectorStoreHandler,self).__init__(args)
        #self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model_name = args["model_name"]

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.embeddings = self.get_embeddings(self.model_name, device)
        #self.embeddings = CustomHuggingFaceEmbeddings(model_name=args["model_name"])
        self.pgconnector = PgVectorConnector()

    @functools.cache
    def get_embeddings(self, model_name, device="cpu"):
        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

    @functools.cache
    def get_vector_store(self, collection_name="coll"):
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=self.pgconnector.get_connection(),
            use_jsonb=True,
        )

        return self.vector_store

    def load_docs_in_vector_store(self, docs):
        """
        docs: list[str] -> [
            {
                "page_content": "...",
                "metadata": {...}
            }
        ]
        """
        start_time = time.time()
        logger.info(f"[{datetime.now()}] Adding {len(docs)} documents into the vector store...")
        hashes = []

        # add model_name to docs metadata
        for i in range(len(docs)):
            docs[i].metadata["model_name"] = self.model_name

        for doc in docs:
            hashed_input = self.hash_doc(doc)
            hashes.append(hashed_input)
        
        conn = self.pgconnector.start_db_connection()
        existing_ids = self.pgconnector.get_existing_ids(conn, hashes, table_name=os.environ['POSTGRES_EMB_TABLE_NAME'])
        self.pgconnector.close_db_connection(conn)
        
        allowed_docs = []
        allowed_hashes = []
        unallowed_docs = [] # for debugging

        for hash, doc in zip(hashes, docs):
            if hash in existing_ids:
                unallowed_docs.append((doc.metadata["source"], doc.metadata["page"]))
            else:
                allowed_docs.append(doc)
                allowed_hashes.append(hash)
        
        if len(unallowed_docs) > 0:
            logger.info(f"[{datetime.now()}] The following documents {unallowed_docs} are already in the db. Skipping...")

        for hash, doc in tqdm(zip(allowed_hashes, allowed_docs)): #processing docs and hashes one-by-one to prevent db connection drops (https://docs.sqlalchemy.org/en/20/errors.html#error-e3q8)
            self.vector_store.add_documents([doc], ids=[hash])
        end_time = time.time()
        logger.info(f"[{datetime.now()}] Added {len(allowed_docs)} documents in {end_time - start_time} seconds")

    def delete_from_vector_store(self, ids: Union[list[str], str], collection_name="coll"):
        start_time = time.time()
        logger.info(f"[{datetime.now()}] Removing {ids} documents from the vector store...")
        if isinstance(ids, str):
            if ids != "all":
                raise ValueError(f"{ids} is not a valid index. Please provide a list of indices or the \"all\" string")
            self.vector_store.delete_collection(collection_name)
        else:
            self.vector_store.delete(ids=ids)
        end_time = time.time()
        logger.info(f"[{datetime.now()}] Removed {ids} documents in {end_time - start_time} seconds")

    @functools.cache
    def query_by_similarity(self, query, k=50, filters=(), with_scores=False):
        d_filter = {}
        for i in range(len(filters)):
            d_filter[filters[i][0]] = filters[i][1]
        
        if with_scores:
            result = self.vector_store.similarity_search_with_score(query, k=k, filter=d_filter)
        else:
            result = self.vector_store.similarity_search(query, k=k, filter=d_filter)
        #print(result[:5])

        return result


class CustomTFIDFRetriever(TFIDFRetriever):

    vectorizer: Any
    """TF-IDF vectorizer."""
    docs: List[Document]
    """Documents."""
    tfidf_array: Any
    """TF-IDF array."""
    k: int = 4
    """Number of documents to return."""

    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform(
            [query]
        )
        results = cosine_similarity(self.tfidf_array, query_vec).reshape(
            (-1,)
        )
        scores = sorted(results)[-self.k :][::-1]
        return_docs = [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
        return return_docs, scores


class SparseStoreHandler(Handler):
    def __init__(self, args):
        super(SparseStoreHandler,self).__init__(args)
        self.switch_model = {
            "tf_idf": CustomTFIDFRetriever
        }
        self.model_name = args["syn_model_name"]
        self.pgconnector = PgVectorConnector()

    def load_docs_in_sparse_store(self, docs):
        start_time = time.time()
        logger.info(f"[{datetime.now()}] Adding {len(docs)} documents into the sparse store...")
        hashes = []

        # add model_name to docs metadata
        for i in range(len(docs)):
            docs[i].metadata["model_name"] = self.model_name

        for doc in docs:
            hashed_input = self.hash_doc(doc)
            hashes.append(hashed_input)
        
        conn = self.pgconnector.start_db_connection()
        existing_ids = self.pgconnector.get_existing_ids(conn, hashes, table_name=os.environ['POSTGRES_SPARSE_TABLE_NAME'])
        
        allowed_docs = []
        allowed_hashes = []
        unallowed_docs = [] # for debugging

        for hash, doc in zip(hashes, docs):
            if hash in existing_ids:
                unallowed_docs.append((doc.metadata["source"], doc.metadata["page"]))
            else:
                allowed_docs.append(doc)
                allowed_hashes.append(hash)

        if len(unallowed_docs) > 0:
            logger.info(f"[{datetime.now()}] The following documents {unallowed_docs} are already in the db. Skipping...")

        for hash, doc in tqdm(zip(allowed_hashes, allowed_docs)): #processing docs and hashes one-by-one to prevent db connection drops (https://docs.sqlalchemy.org/en/20/errors.html#error-e3q8)
            elements_to_add = (
                hash,
                '.'.join(doc.metadata["source"].split("/")[-1].split(".")[:-1]),
                doc.metadata["source"],
                doc.page_content,
                doc.metadata["page"],
                self.model_name
            )

            self.pgconnector.add_page(conn, elements_to_add)

        self.pgconnector.close_db_connection(conn)
        end_time = time.time()
        logger.info(f"[{datetime.now()}] Added {len(allowed_docs)} documents in {end_time - start_time} seconds")

    @functools.cache
    def query_by_similarity(self, query, source, k=50, with_scores=False):
        conn = self.pgconnector.start_db_connection()
        docs, docs_lowered = self.pgconnector.get_pages(conn, source)
        self.pgconnector.close_db_connection(conn)

        retriever = self.switch_model[self.model_name].from_documents(docs_lowered)

        retriever.k = k
        results, scores = retriever.get_relevant_documents(query)

        if with_scores:
            result_with_scores = []
            for result, score in zip(results, scores):
                result_with_scores.append((result, score))
            results = result_with_scores

        return results

class EnsembleRetrieverHandler(SparseStoreHandler, VectorStoreHandler):
    def __init__(self, args):
        super(EnsembleRetrieverHandler, self).__init__(args)
        self.lmbd = args["lambda"]

    def combine_results(self, semantic_results, syntactic_results, k=50, lmbd = .3):
        results = {}
        res_debug = {}
        for i,r in enumerate([semantic_results, syntactic_results]):
            for j,el in enumerate(r):

                # semantic and syntactic Documents have different model_names
                # we change them to an empty string to allow Document matching

                el[0].metadata["model_name"] = ""
                if i == 0:
                    el[0].page_content = el[0].page_content.lower()
                
                hashed_key = self.hash_doc(el[0])
                score = -el[1] if i == 0 else el[1]*lmbd
                if hashed_key not in results.keys():
                    results[hashed_key] = [el[0], score]
                    res_debug[hashed_key] = [score, 0]
                else:
                    results[hashed_key][1] += score
                    res_debug[hashed_key][1] = score
        
        values = results.values()
        values = sorted(values, key=lambda x: x[1], reverse=True)[:k]

        return [el[0] for el in values]

    @functools.cache
    def query_by_similarity(self, query, filters, k=50):
        if len(filters) == 0:
            raise ValueError(f"\"filters\" param cannot be empty")
        if filters[0][0] != "source":
            raise ValueError(f"filter \"source\" is expected in first position of filters param")
        if not filters[0][1] or not isinstance(filters[0][0], str):
            raise ValueError(f"the value of the \"source\" param must be a string")

        self.get_vector_store()
        semantic_results = VectorStoreHandler.query_by_similarity(self, query, k=1000, filters=filters, with_scores=True)
        syntactic_results = SparseStoreHandler.query_by_similarity(self, query, source=filters[0][1], k=1000, with_scores=True)

        results = self.combine_results(semantic_results, syntactic_results, k=k, lmbd=self.lmbd)

        return results

