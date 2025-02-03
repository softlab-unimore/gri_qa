import functools
import logging
import time
import os

from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.dataprocessor")

class PageProcessor:
    """
    This class extracts textual contents from pdf files.
    It is used when trying to do RAG on simple pages, without any specific image/table handling
    """

    def __init__(self):
        pass

    @functools.cache
    def _get_reader(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise ValueError(f"{pdf_path} is non existent")
        
        loader = PyPDFLoader(pdf_path)
        try:
           data = loader.load_and_split()
        except TypeError as e:
            logger.warning(f"Chunking failed for {pdf_path} with the following error: {e}")
            data = []
        return data

    @functools.cache
    def get_pdf_content(self, pdf_path):
        data = []
        if os.path.isdir(pdf_path):
            files = os.listdir(pdf_path)
            start_time = time.time()
            logger.info(f"[{datetime.now()}] Chunking {len(files)} files...")
            for file in tqdm(files):
                if file.split(".")[-1] != "pdf":
                    continue
                docs = self._get_reader(os.path.join(pdf_path, file))
                data.extend(docs)
            end_time = time.time()
            logger.info(f"[{datetime.now()}] Chunked {len(files)} files in {end_time - start_time} seconds")
        else:
            docs = self._get_reader(pdf_path)
            data.extend(docs)

        return data
    
