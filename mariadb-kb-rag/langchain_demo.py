from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_mariadb import MariaDBStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy import create_engine
from sqlalchemy import text
from langchain_openai import ChatOpenAI
from langchain.chains.sql_database.prompt import SQL_PROMPTS
from langchain_core.output_parsers.string import StrOutputParser
from operator import itemgetter
from collections import defaultdict
from tabulate import tabulate
from langchain.globals import set_verbose
from langchain.globals import set_debug
import argparse

# Create argument parser
parser = argparse.ArgumentParser(description="MariaDB RAG for Text2SQL")


# Add -v (flag, no value)
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
parser.add_argument('-d', '--debug', action='store_true', help='Enable verbose mode')
parser.add_argument('-n', '--nocontext', action='store_true', help='Enable verbose mode')

# Add -p [prompt] (option with value)
parser.add_argument('-p', '--prompt', type=str, help='Prompt message')

# Parse args
args = parser.parse_args()
if args.debug:
    set_debug(True)
elif args.verbose:
    set_verbose(True)
import os
import mariadb
import traceback

uri = "mariadb+mariadbconnector://root:Password123!@127.0.0.1:3306/kb_rag"
engine = create_engine(uri)
openai_api_key=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, api_key=openai_api_key, model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
conn = mariadb.connect(
    host="127.0.0.1", port=3306, user="root", password="Password123!"
)
cur = conn.cursor()
cur.execute("USE sakila")
cur.execute("ROLLBACK;")
cur.execute("SET autocommit = 0;")

def get_results(sql_query):
    for statement in sql_query.strip().split(';'):
        if statement.strip():
            cur.execute(statement)
    results = list(cur)
    print(sql_query)
    print(results)
    return results

def list_tables():
    results = get_results("SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA='sakila';")
    return [x[0] for x in results]

def list_views():
    results = get_results("SELECT TABLE_NAME FROM information_schema.VIEWS WHERE TABLE_SCHEMA='sakila';")
    return [x[0] for x in results]

# TODO make this called before every run
all_tables = list_tables()
all_views = list_views()
all_tables_and_views = set(all_tables).union(set(all_views))


def get_table_info(table_names, query):
    # Safely create the placeholders and params
    if len(table_names) == 0:
        return get_table_info(list(all_tables_and_views), query)
    placeholders = ', '.join(['%s'] * len(table_names))
    query = f"""SELECT
    table_name,
    column_name,
    column_type,
    is_nullable,
    column_default,
    extra,
    column_key
    FROM information_schema.columns
    WHERE table_schema = %s
    AND table_name IN ({placeholders})"""

    # First param is the schema name, followed by table names
    if not type(table_names[0]) is str:
        table_names = [x.name for x in table_names]
    params = ['sakila'] + table_names
    print(table_names)
    print(all_tables)
    cur.execute(query, params)
    results = cur.fetchall()
    if len(results) == 0:
        # TODO handle the case where it disobeys orders or misnames
        return get_table_info(list(all_tables_and_views), query)
    # Organize by table
    schema = defaultdict(list)

    for table, column, col_type, nullable, default, extra, key in results:
        schema[table].append({
            "Column": column,
            "Type": col_type,
            "Nullable": nullable == 'YES',
            "Default": default,
            "Extra": extra,
            "Key": key
        })
    # Pretty print tables
    rtn = ""
    for table_name, columns in schema.items():
        if len(columns) > 20:
            columns = ask_row_agent(table_name, columns, query)
        rtn += f"\n Table: {table_name}\n"
        rtn += str(tabulate(columns, headers="keys", tablefmt="pretty"))+"\n"
    print(rtn)
    return rtn

# annoying workaround to a currently-broken feature
def _strip(text: str) -> str:
    return text.strip()

def create_sql_query_chain(
    llm,
    cur,
    prompt = None,
    k: int = 5,
):
    if prompt is not None:
        prompt_to_use = prompt
    else:
        prompt_to_use = SQL_PROMPTS['mariadb']

    if {"input", "top_k", "table_info"}.difference(
        prompt_to_use.input_variables + list(prompt_to_use.partial_variables)
    ):
        raise ValueError(
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )
    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect='mariadb')

    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "table_info": lambda x: get_table_info(
            table_names=x.get("table_names_to_use"), query=x["question"]
        ),
        "kb_entry": lambda x: search_for_closest_kb(x["question"])[0]['content'],
        "similar_queries": lambda x: search_for_relevant_queries(x["question"]),
    }

    return (
        RunnablePassthrough.assign(**inputs)  # type: ignore
        | (
            lambda x: {
                k: v
                for k, v in x.items()
                if k not in ("question", "table_names_to_use")
            }
        )
        | prompt_to_use.partial(top_k=str(k))
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
        | _strip
    )

# unfortunate, but for now we have to have 2 connections because of discrepancies between
# sources of the vectors (we could use the metadata, but it conflicts with how we loaded kb_content)
vectorstore_query = MariaDBStore(
    embeddings=embeddings,
    embedding_length=1536,
    datasource=uri,
    collection_name="query_rag",
)


def search_for_relevant_queries(text):
    docs = vectorstore_query.max_marginal_relevance_search(text)
    rtn = ""
    for doc in docs:
        rtn += doc.page_content + "\n"
    return rtn

def search_for_closest_kb(text, n=1):
    embedding = embeddings.embed_query(text)
    cur.execute(
        f"""
        SELECT title, url, content,
               VEC_DISTANCE_EUCLIDEAN(embedding, VEC_FromText(%s)) AS distance
        FROM kb_rag.kb_content
        ORDER BY distance ASC
        LIMIT %s;
    """,
        (str(embedding), n),
    )

    closest_content = [
        {"title": title, "url": url, "content": content, "distance": distance}
        for title, url, content, distance in cur
    ]
    return closest_content


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")

def ask_table_agent(query):
    table_names = "\n".join(all_tables)
    view_names = "\n".join(all_views)
    system = f"""Return the names of ALL the SQL tables and views that MIGHT be relevant to the user question. \
    The tables are:
    {table_names}

    The views are:
    {view_names}

    Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed. Return at least one table or view."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{input}"),
        ]
    )
    llm_with_tools = llm.bind_tools([Table])
    output_parser = PydanticToolsParser(tools=[Table])

    table_chain = prompt | llm_with_tools | output_parser
    return table_chain

def ask_row_agent(table_name, columns, query):
    # TODO: rewrite this
    system = f"""Return the names of ALL the columns from the given table that MIGHT be relevant to the user question. \
    The table name is: {table_name}
    The columns are: {columns}

    Remember to include ALL POTENTIALLY RELEVANT names, even if you're not sure that they're needed."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{input}"),
        ]
    )
    return prompt

def build_query_components(query):
    table_chain = ask_table_agent(query)

    previous_attempts = ""

    system = """You are a MariaDB expert. Given an input question, create a syntactically
correct MariaDB query to run. Unless otherwise specificed, do not return more than
{top_k} rows.

Only return the SQL query with no markup or explanation.

Here is the relevant table info: {table_info}

Here is the relevant knowledge about MariaDB for this type of query: {kb_entry}

Here are the relevant similar queries: {similar_queries}"""
    system_nocontext = """You are a MariaDB expert. Given an input question, create a syntactically
correct MariaDB query to run. Unless otherwise specificed, do not return more than
{top_k} rows.

Only return the SQL query with no markup or explanation. {table_info}"""
    if args.nocontext:
        system = system_nocontext
    # Convert "question" key to the "input" key expected by current table_chain.
    table_chain = {"input": itemgetter("question")} | table_chain
    for tries in range(3):
        prompt = ChatPromptTemplate.from_messages([("system", system + previous_attempts), ("human", "{input}")])
        query_chain = create_sql_query_chain(llm, cur, prompt=prompt)

        # Set table_names_to_use using table_chain.
        full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain
        q = full_chain.invoke({"question": query})
        print(q)
        q = '\n'.join(line for line in q.splitlines() if not line.strip().startswith('```') )
        try:
            get_results(q)
            break
        except Exception as e:
            print("Error: ", e)
            previous_attempts += f"\nLast time we tried {q}\n This errored with " + str(e)


def use_kb_rag(db, fn):
    # handle annoying database error because of underscore in table name
    cur.execute("USE kb_rag")
    rtn = fn()
    cur.execute("USE sakila")
    return rtn

if args.prompt:
    build_query_components(args.prompt)
else:
    build_query_components("How many customers and employees are there in each country?")
