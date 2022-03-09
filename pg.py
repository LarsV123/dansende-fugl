import csv
import re
import click
import os
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm
from rich import print
from itertools import islice
from utils import time_this
from dotenv import load_dotenv, find_dotenv


# Load database credentials into environment variables
load_dotenv(find_dotenv())

# Increase max field size to handle huge text fields in CSV files
csv.field_size_limit(0x2000000)

# Paths to all .CSV and .SQL files
scripts = {
    "typed_posts": {
        "data": "data/typed_posts.csv",
        "schema": "schema/typed_posts.sql",
        "index": "schema/typed_posts_indexes.sql",
        "view": "schema/view_posts.sql",
    },
    "typed_comments": {
        "data": "data/typed_comments.csv",
        "schema": "schema/typed_comments.sql",
        "index": "schema/typed_comments_indexes.sql",
        "view": "schema/view_comments.sql",
    },
    "mbti9k": {
        "data": "data/mbti9k_comments.csv",
        "schema": "schema/mbti9k.sql",
        "index": "schema/mbti9k_indexes.sql"
    }
}


class Connector:
    """
    Connects to the Postgres server, using credentials stored in
    environment variables.
    """

    def __init__(self, verbose=True):
        # Toggle whether or not debug print statements are used
        self.verbose = verbose

        # Connect to the Postgres server
        self.connection = psycopg2.connect(
            host=os.environ.get("HOST"),
            database=os.environ.get("POSTGRES_DB"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
        )

        # Create a cursor
        self.cursor = self.connection.cursor()

        # Check connection
        if self.verbose:
            self.cursor.execute("SELECT version()")
            db_version = self.cursor.fetchone()
            print(f"Connected to: {db_version[0]}")

    def execute_script(self, path: str):
        """
        Execute script stored as .sql file
        """
        with open(path, "r") as file:
            self.cursor.execute(file.read())

    def vacuum_analyze(self):
        """
        This needs to run after a bulk load of data.
        """
        old_isolation_level = self.connection.isolation_level
        self.connection.set_isolation_level(0)
        query = "VACUUM ANALYZE"
        self.cursor.execute(query)
        self.connection.set_isolation_level(old_isolation_level)

    def close(self):
        self.cursor.close()
        self.connection.close()
        if self.verbose:
            print("Connection to Postgres database closed")

    def has_table(self, table: str):
        query = f"""
        --sql
        SELECT EXISTS (
            SELECT FROM 
                pg_tables
            WHERE 
                schemaname = 'public' AND 
                tablename  = '{table}'
            )
        ;
        """
        self.cursor.execute(query)
        exists = self.cursor.fetchone()[0]
        return exists

    @time_this
    def initialize_table(self, table: str):
        """
        Helper-method for initializing a table with indexes and views,
        optionally dropping and recreating schema.
        """
        exists = self.has_table(table)
        if exists:
            # Check if the user actually wants to drop the table
            row_query = f"SELECT COUNT(*) FROM {table};"
            self.cursor.execute(row_query)
            row_count = self.cursor.fetchone()[0]
            if row_count > 0:
                print(
                    f"[bold red]WARNING:[/bold red] {table} contains {row_count} rows"
                )

        init = click.confirm(f"Do you want to drop and recreate the table '{table}'?")

        if init or not exists:
            print(f"Initializing schema for '{table}'...")
            self.execute_script(scripts[table]["schema"])
            self.connection.commit()

            if table not in scripts:
                raise ValueError

            if table == "typed_posts":
                insert_posts(self, table)
            elif table == "typed_comments":
                insert_csv(self, table, 25000)
            elif table == "mbti9k":
                insert_csv(self, table, 25)
            else:
                raise ValueError
            print()

        # Safe to do this anyway
        if "index" in scripts[table]:
            print(f"Creating indexes for {table}...")
            self.execute_script(scripts[table]["index"])

        if "view" in scripts[table]:
            print(f"Creating views for {table}...")
            self.execute_script(scripts[table]["view"])

        self.connection.commit()
        print(f"Finished initializing {table}")


def parse_posts(path: str) -> list:
    """
    Read file, split on correct line endings and return list of rows.

    Because the CSV file is malformed (unquoted text fields with linebreaks), we need to parse it manually.
    """
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

        # Split on true line ending (comma, 4 characters and a newline)
        pattern = r"(?<=,[a-z]{4})\n"
        text = re.split(pattern, text)

        # Remove all newline characters and empty rows
        text = [x.replace("\n", "") for x in text if len(x) > 0]
    return text


def insert_posts(db: Connector, table: str):
    """
    Insert all rows from typed_posts.csv into Postgres.
    """
    path = scripts[table]["data"]
    data = parse_posts(path)
    csv_reader = csv.reader(data)
    next(csv_reader)  # Discard header
    query = f"INSERT INTO {table} VALUES %s;"

    all_rows = list(csv_reader)
    total_rows = len(all_rows)
    n = 50000  # Rows per transaction
    print(f"Inserting {total_rows} rows into table {table}")

    progress_bar = tqdm(total=total_rows)
    for i in range(0, total_rows, n):
        batch = all_rows[i : i + n]
        execute_values(db.cursor, query, batch, page_size=n)
        db.connection.commit()
        progress_bar.update(n)
    progress_bar.close()

    print(f"Finished inserting {total_rows} rows into table '{table}'")


def insert_csv(db: Connector, table: str, n: int):
    """
    Insert all rows from the specified table into Postgres.
    """
    total_rows = 0
    query = f"INSERT INTO {table} VALUES %s;"

    path = scripts[table]["data"]
    file = open(path, encoding="utf8")
    file.readline()  # Discard header
    print("discarded header for", table)

    progress_bar = tqdm()
    for chunk in iter(lambda: tuple(islice(file, n)), ()):
        # Parse
        csv_reader = csv.reader(chunk)
        batch = list(csv_reader)
        line_count = len(batch)

        # Write to database
        execute_values(db.cursor, query, batch, page_size=n)

        # Update statistics
        total_rows += line_count
        progress_bar.update(line_count)
        db.connection.commit()

    file.close()
    progress_bar.close()
    print(f"Finished inserting {total_rows} rows into table '{table}'")
    print()
