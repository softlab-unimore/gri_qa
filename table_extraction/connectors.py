import os
import logging
import psycopg2

from langchain_core.documents import Document

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.connector")

class PgVectorConnector:
    def __init__(self):
        username = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
        db = os.environ["POSTGRES_DB"]
        port = os.environ["POSTGRES_PORT"]
        self.connection = f"postgresql+psycopg://{username}:{password}@localhost:{port}/{db}"
    
    def get_connection(self):
        return self.connection
    
    def start_db_connection(self):
        connection = self.connection.replace("+psycopg", "")
        return psycopg2.connect(connection)

    def close_db_connection(self, conn):
        conn.close()

    def get_existing_ids(self, conn, ids, table_name):
        formatted_ids = ', '.join(f"'{id}'" for id in ids)
        query = f"SELECT id FROM {table_name} WHERE id IN ({formatted_ids});"

        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
        return [row[0] for row in result]

    def add_page(self, conn, elements_to_add: tuple):
        if len(elements_to_add) != 6:
            raise ValueError(f"the new row to add in the \"{os.environ['POSTGRES_SPARSE_TABLE_NAME']}\" table must have 4 elements")
        #if not all([isinstance(el, str) for el in elements_to_add]):
            #raise ValueError(f"the new row to add in the \"{os.environ['POSTGRES_SPARSE_TABLE_NAME']}\" table can only have str elements")

        query = f"INSERT INTO {os.environ['POSTGRES_SPARSE_TABLE_NAME']} (id, title, source, page_content, page_nbr, model_name) VALUES(%s, %s, %s, %s, %s, %s);"

        with conn.cursor() as cur:
            cur.execute(query, elements_to_add)
            a = conn.commit()

    def get_pages(self, conn, source):
        query = f"SELECT source, page_nbr, model_name, page_content FROM {os.environ['POSTGRES_SPARSE_TABLE_NAME']} WHERE source=%s;"

        print(source)
        with conn.cursor() as cur:
            cur.execute(query, (source,))
            result = cur.fetchall()
        print(result)
        docs = []
        docs_lowered = []
        for res in result:
            doc = Document(page_content=res[-1], metadata={"page": res[1], "source": res[0], "model_name": res[2]})
            doc_lowered = Document(page_content=res[-1].lower(), metadata={"page": res[1], "source": res[0], "model_name": res[2]})
            docs.append(doc)
            docs_lowered.append(doc_lowered)
        
        return docs, docs_lowered
        