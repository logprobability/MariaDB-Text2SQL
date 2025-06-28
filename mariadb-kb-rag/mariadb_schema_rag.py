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
