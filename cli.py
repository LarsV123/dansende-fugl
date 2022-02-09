from email.policy import default
from rich import print
import click
import pandas as pd
import sqlite3

from queries import get_user_data

database = "data.db"
tables = ["typed_posts", "typed_comments", "mbti9k_comments"]
connection = sqlite3.connect(database)
cursor = connection.cursor()


@click.group()
def cli():
    pass


@cli.command()
@click.option("--count", default=1, help="Number of greetings")
@click.argument("name")
def hello(count, name):
    for _ in range(count):
        print(f"Hello [bold red]{name}[/bold red]!")


def load_dataset_to_sql(dataset="typed_posts", folder="./data"):
    path = f"{folder}/{dataset}.csv"

    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)

    print(f"Writing dataset to table {dataset} in {database}")
    df.to_sql(dataset, connection, if_exists="replace", index=False)
    print()


@cli.command()
def init():
    for table in tables:
        load_dataset_to_sql(dataset=table)
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


@cli.command()
@click.option(
    "--table",
    type=click.Choice(tables),
    default="mbti9k_comments",
    help="Table to fetch from",
)
@click.option("--user", help="User to fetch data about")
def get_user(table, user):
    rows = get_user_data(connection, table, user)
    for row in rows:
        print(row)


if __name__ == "__main__":
    cli()
