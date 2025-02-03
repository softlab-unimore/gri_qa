import logging
import time
import traceback

from dataprocessor import PageProcessor
from vector_store import VectorStoreHandler, SparseStoreHandler, EnsembleRetrieverHandler

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.main")

class Runnable:

    def __init__(self, args):
        self.switch_method = {
            "page": PageProcessor,
        }

        if args["use_ensemble"]:
            self.ens = EnsembleRetrieverHandler(args)
        else:
            self.vsh = VectorStoreHandler(args)
            self.ssh = SparseStoreHandler(args)

        try:
            self.processor = self.switch_method[args["method"]]()
        except:
            raise ValueError(f"{args['method']} is not a valid extraction method")

        self.args = args

    def set_args(self, args):
        self.args = args

    def run(self):
        if self.args["use_dense"]:
            self.vsh.get_vector_store()

        contents = self.processor.get_pdf_content(self.args["pdf"])

        if self.args["embed"]:
            if self.args["use_dense"]:
                self.vsh.load_docs_in_vector_store(contents)
            elif self.args["use_sparse"]:
                self.ssh.load_docs_in_sparse_store(contents)
            similar_docs = None
        elif self.args["query"]:
            filters = (("source", self.args["pdf"]), ("model_name", self.args["model_name"]))
            if self.args["use_dense"]:
                similar_docs = self.vsh.query_by_similarity(self.args["query"], filters=filters)
            elif self.args["use_sparse"]:
                similar_docs = self.ssh.query_by_similarity(self.args["query"], source=self.args["pdf"])
            elif self.args["use_ensemble"]:
                similar_docs = self.ens.query_by_similarity(self.args["query"], filters=filters)

        return similar_docs