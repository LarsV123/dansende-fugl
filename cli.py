from rich import print
from tqdm import tqdm
import click
import pandas as pd
import sqlite3

database = "data.db"
tables = ["typed_posts", "typed_comments", "mbti9k_comments"]


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
    connection = sqlite3.connect(database)
    # from https://stackoverflow.com/a/39495229
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


@cli.command()
def init():
    folder = "./data"
    for table in tables:
        path = f"{folder}/{table}.csv"
        print(f"Loading dataset: {path}")
        df = pd.read_csv(path)

        drop_table(table)
        print(f"Writing dataset to table {table} in {database}")

        insert_with_progress(df, table)
        print()

    print(f"Successfully loaded {len(tables)} files into SQLite")


@cli.command()
@click.option("--count", default=2, help="Number of rows to fetch")
@click.option("--table", type=click.Choice(tables), help="Table to sample")
def sample(table, count):
    query = f"""
    --sql
    SELECT * FROM {table}
    LIMIT {count}
    ;
    """
    rows = cursor.execute(query).fetchall()
    for row in rows:
        print(row)


if __name__ == "__main__":
    cli()
