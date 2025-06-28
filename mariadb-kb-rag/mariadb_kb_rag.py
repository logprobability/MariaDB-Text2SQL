# Demo of RAG with MariaDB Vector using content from the MariaDB Knowledge Base

import mariadb
from openai import OpenAI
import json
import os

# To run MariaDB 11.7.1 with Docker:
# docker run -p 127.0.0.1:3306:3306  --name mdb_117 -e MARIADB_ROOT_PASSWORD=Password123! -d mariadb:11.7-rc

# For embedding, OpenAI API key is needed. Create at https://platform.openai.com/api-keys and add to system variables with export OPENAI_API_KEY='your-key-here'.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conn = mariadb.connect(
    host="127.0.0.1", port=3306, user="root", password="Password123!"
)
cur = conn.cursor()


# prepare_database()


def read_kb_from_file(filename):
    with open(filename, "r") as file:
        return [json.loads(line) for line in file]


# chunkify by paragraphs, headers, etc.
def chunkify(content, min_chars=1000, max_chars=10000):
    lines = content.split("\n")
    chunks, chunk, length, start = [], [], 0, 0
    for i, line in enumerate(lines + [""]):
        if (
            chunk
            and (
                line.lstrip().startswith("#")
                or not line.strip()
                or length + len(line) > max_chars
            )
            and length >= min_chars
        ):
            chunks.append(
                {
                    "content": "\n".join(chunk).strip(),
                    "start_line": start,
                    "end_line": i - 1,
                }
            )
            chunk, length, start = [], 0, i
        chunk.append(line)
        length += len(line) + 1
    return chunks


def embed(text):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",  # max 8192 tokens (roughly 32k chars)
        )
    except:
        return None
    return response.data[0].embedding


def insert_kb_into_db():
    kb_pages = read_kb_from_file("kb_scraped_md_full.jsonl") # alternatively kb_scraped_md_full.jsonl
    for p in kb_pages:
        if ('meta' in p["url"]) or ('changelog' in p["url"]) or ('release-notes' in p["url"]) or ('.' in p["title"]) or ('InnoDB' in p["title"]) or ('S3' in p["title"]) or (p["title"][0] == ' ') or ('Amazon' in p["title"]) or ('Google' in p["title"]):
            continue
        chunks = chunkify(p["content"])
        for chunk in chunks:
            chunklength = len(chunk["content"])
            p_title = p["title"]
            print(
                f"Embedding chunk (length {chunklength}) from '{p_title}'"
            )
            embedding = embed(chunk["content"])
            if embedding:
                cur.execute(
                    """INSERT INTO kb_rag.kb_content (title, url, content, embedding)
                        VALUES (%s, %s, %s, VEC_FromText(%s))""",
                    (p["title"], p["url"], chunk["content"], str(embedding)),
                )
        conn.commit()


# insert_kb_into_db()


def search_for_closest_content(text, n, table="kb_content"):
    embedding = embed(text)  # using same embedding model as in preparations
    cur.execute(
        f"""
        SELECT title, url, content, 
               VEC_DISTANCE_EUCLIDEAN(embedding, VEC_FromText(%s)) AS distance
        FROM kb_rag.{table}
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


user_input = "How do I store compressed data using MariaDB?"
print(f"Prompting:\n'{user_input}'")
closest_content = search_for_closest_content(user_input, 2)


def prompt_chat(system_prompt, prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


system_prompt_with_rag = """
You are a helpful assistant that answers questions using exclusively content from the MariaDB Knowledge Base that you are provided with. 
End your answer with a link to the most relevant content given to you.
"""
prompt_with_rag = f""" 
    The user asked: '{user_input}'.  
    Relevant content from the MariaDB Knowledge Base:
    '{str(closest_content)}'
    """
print(prompt_with_rag)
print(
    f"""
    LLM response with RAG:'
    {prompt_chat(system_prompt_with_rag,prompt_with_rag)}'
    """
)

system_prompt_no_rag = """
You are a helpful assistant that only answers questions about MariaDB. 
End your answer with a link to a relevant source.
"""
prompt_no_rag = f""" 
   The user asked: '{user_input}'.
   """
print(
    f"""
   LLM response without RAG:
   '{prompt_chat(system_prompt_no_rag, prompt_no_rag)}'
   """
)
