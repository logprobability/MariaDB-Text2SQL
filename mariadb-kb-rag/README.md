# Demo: RAG with MariaDB Vector

A Python script that implements Retrieval-Augmented Generation (RAG) on the MariaDB Knowledge Base using MariaDB's vector database capabilities, OpenAI's embedding model, and OpenAI's LLM.

## Features

1. Reads MariaDB Knowledge Base articles from a JSONL file
2. Splits articles into manageable "chunks"
3. Creates vectors (embeddings) for each chunk using OpenAI's embedding model
4. Inserts content and vectors into MariaDB
5. Takes a user input and vectorizes it
6. Searches for most relevant chunks in MariaDB using nearest neighbor search
7. Generates response for user by prompting OpenAI's LLM with user input and relevant chunks

## Prerequisites

- Docker
- Python 3.x (developed with 3.13)
- OpenAI API key
- MariaDB 11.7.1 or later

## Setup

1. Start MariaDB 11.7.1 with Docker: 

```bash
docker run -p 127.0.0.1:3306:3306  --name mdb_117 -e MARIADB_ROOT_PASSWORD=Password123! -d mariadb:11.7-rc
```

If needed, access MariaDB with:

```bash
docker exec -it mdb_1171 mariadb --user root -pPassword123!
```

If needed, install Docker with:

```bash
brew install docker
```

2. Add your OpenAI API key to your environment variables. Create a key at https://platform.openai.com/api-keys. 

```bash
export OPENAI_API_KEY='your-key-here'
```

3. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

4. Install the required Python packages with pip

```bash
pip install -r requirements.txt
```

## Usage

Running the following  at the moment does everything from setting up the database, to printing a LLM response to a question.

```bash
python mariadb_kb_rag.py
```

## Possible ways to develop further

- Separate preparation and running questions to own python files
- Test other embedding and LLM models
- Add a graphical GUI with flask, vue, streamlit, gradio, etc.
- Add timer to see how long database operations vs OpenAI calls take (>1s for OpenAI calls, less than 0.1s for database operations)
- Try out more advanced chunking solutions, like https://github.com/bhavnicksm/chonkie/

## Resources used to put this together

- Cursor https://www.cursor.com/
- MariaDB Vector https://mariadb.com/kb/en/vector-overview/
- Installing and Using MariaDB via Docker https://mariadb.com/kb/en/installing-and-using-mariadb-via-docker/
- OpenAI Embeddings https://platform.openai.com/docs/guides/embeddings
- OpenAI Chat Completions https://platform.openai.com/docs/guides/text-generation