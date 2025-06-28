import mariadb
import sys


# 📂 Path to your SQL file
sql_file_path = 'sakila-schema.sql' #'mysql-sakila-schema.sql'

try:
    conn = mariadb.connect(
        host="127.0.0.1", port=3306, user="root", password="Password123!"
    )
    cur = conn.cursor()

    # 🧬 Connect to MariaDB

    # 📖 Read the SQL file
    with open(sql_file_path, 'r') as file:
        sql_script = file.read()

    # 🧾 Split and execute each statement
    for statement in sql_script.split(';'):
        stmt = statement.strip()
        if stmt:
            cur.execute(stmt)

    conn.commit()
    print("✅ SQL script executed successfully.")

except mariadb.Error as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

finally:
    if conn:
        conn.close()
