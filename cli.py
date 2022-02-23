import click
import os
import pandas as pd
import sqlite3
from rich import print
from tqdm import tqdm
from queries import get_user_data

from queries import get_user_data

database = "data.db"
tables = ["typed_posts", "typed_comments", "mbti9k_comments"]
data_folder = "./data"


@click.group()
def cli():
    pass


@cli.command()
@click.option("--count", default=1, help="Number of greetings")
@click.argument("name")
def hello(count, name):
    for _ in range(count):
        print(f"Hello [bold red]{name}[/bold red]!")


def chunker(seq, size):
    # from http://stackoverflow.com/a/434328
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def insert_with_progress(df, table: str):
    # from https://stackoverflow.com/a/39495229
    connection = sqlite3.connect(database)
    chunksize = int(len(df) / 100)  # 1%
    with tqdm(total=len(df)) as pbar:
        for i, cdf in enumerate(chunker(df, chunksize)):
            replace = "replace" if i == 0 else "append"
            cdf.to_sql(con=connection, name=table, if_exists=replace, index=False)
            pbar.update(chunksize)
    connection.close()


def drop_table(table: str):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    query = f"""
    --sql
    DROP TABLE IF EXISTS {table}
    ;
    """
    cursor.execute(query)
    connection.close()


def execute_script(filename):
    """
    Read and execute line by line a script stored as a .sql file.
    For executing the entire script in one go, use cursor.executescript()
    """
    connection = sqlite3.connect(database)
    cursor = connection.cursor()

    # Open and read the file as a single buffer
    print(f"Executing script: {filename}")
    file = open(filename, "r")

    # Read lines and strip line breaks
    lines = [x.strip().replace("\n", "") for x in file.read().split(";")]

    # Remove empty lines
    sql = [x for x in lines if x]
    file.close()

    # Execute every command from the input file
    for command in sql:
        print()
        # This will skip and report errors
        try:
            print("Executing command:")
            print(command)
            cursor.execute(command)
        except Exception as e:
            print("Command skipped: ", command)
            print("Error:", e)
    connection.close()


@cli.command()
def index():
    """
    Create all relevant indexes using stored SQL scripts.
    """
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    folder = "./scripts/indexes"

    print(f"Running all scripts in {folder}")
    for (root, dirs, files) in os.walk(folder, topdown=True):
        for filename in files:
            path = f"{folder}/{filename}"
            with open(path) as script:
                print(f"Executing script {filename}")
                cursor.executescript(script.read())
    
    print()
    print("Finished creating indexes!")

@cli.command()
def init():
    for (root, dirs, files) in os.walk(data_folder, topdown=True):
        print(files)
        for fn in files:
            table = fn.split(".")[0]
            path = f"{data_folder}/{fn}"
            print(f"Loading dataset: {path}")
            df = pd.read_csv(path)

            drop_table(table)
            print(f"Writing dataset to table {table} in {database}")

            insert_with_progress(df, table)
            print()

    print(f"Successfully loaded {len(files)} files into SQLite")


@cli.command()
@click.option("--count", default=2, help="Number of rows to fetch")
@click.option("--table", type=click.Choice(tables), help="Table to sample")
def sample(table, count):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    query = f"""
    --sql
    SELECT * FROM {table}
    LIMIT {count}
    ;
    """
    rows = cursor.execute(query).fetchall()
    for row in rows:
        print(row)


@cli.command()
@click.option(
    "--table",
    type=click.Choice(tables),
    default="mbti9k_comments",
    help="Table to fetch from",
)
@click.option("--user", help="User to fetch data about")
def get_user(table, user):
    connection = sqlite3.connect(database)
    rows = get_user_data(connection, table, user)
    for row in rows:
        print(row)


if __name__ == "__main__":
    cli()
