from rich import print
import click


@click.group()
def cli():
    pass


@cli.command()
@click.option("--count", default=1, help="Number of greetings")
@click.argument("name")
def hello(count, name):
    for _ in range(count):
        print(f"Hello [bold red]{name}[/bold red]!")


if __name__ == "__main__":
    # Example terminal command:
    # python example.py hello Bob --count 5
    cli()
