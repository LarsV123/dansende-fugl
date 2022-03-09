import click
import pg


@click.group()
def cli():
    pass


@cli.command()
def init():
    db = pg.Connector()
    db.initialize_table("typed_posts")
    db.initialize_table("typed_comments")
    db.vacuum_analyze()
    db.close()


if __name__ == "__main__":
    cli()
