from langchain.llms import OpenAI
from langchain_mariadb import MariaDBStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import traceback
import os
import json

uri = "mariadb+mariadbconnector://root:Password123!@127.0.0.1:3306/kb_rag"
# engine = create_engine(uri)
openai_api_key=os.getenv("OPENAI_API_KEY")

vectorstore_query = MariaDBStore(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    embedding_length=1536,
    datasource=uri,
    collection_name="query_rag",
)

# vectorstore_query.drop_tables()

def read_kb_from_file(filename):
    with open(filename, "r") as file:
        return [json.loads(line) for line in file]

def embed_queries(q, category):
    texts = [x['description'].replace('\\n', '\n')+'\n\n'+x['query'].replace('\\n', '\n') for x in q]
    metadatas = [{'category': category} for _ in texts]
    vectorstore_query.add_texts(texts=texts, metadatas=metadatas)

# set up the example query rag
def get_example_queries():
    three_examples = read_kb_from_file('three_queries.jsonl')
    embed_queries(three_examples, 'three_examples')
    sql_tasks = read_kb_from_file('sql_tasks.jsonl')
    embed_queries(sql_tasks, 'tasks')
    advanced_queries = read_kb_from_file('sakila_advanced_queries.jsonl')
    embed_queries(advanced_queries, 'advanced')

get_example_queries()


