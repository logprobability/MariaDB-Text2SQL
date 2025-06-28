
def prepare_database():
    print("Delete database")
    cur.execute("""DROP DATABASE IF EXISTS kb_rag;""")

    print("Create database and table")
    cur.execute(
        """
        CREATE DATABASE IF NOT EXISTS kb_rag;
        """
    )
    cur.execute(
        """
        CREATE TABLE kb_rag.kb_content (
            title VARCHAR(255) NOT NULL,
            url VARCHAR(255) NOT NULL,
            content LONGTEXT NOT NULL,
            embedding VECTOR(1536) NOT NULL,
            VECTOR INDEX (embedding)
        );
        """
    )
