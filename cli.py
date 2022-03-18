import click
import pg


@click.group()
def cli():
    pass


@cli.command()
def init():
    """
    Run once to initialize database from CSV files.
    """
    db = pg.Connector()
    db.initialize_table("typed_posts")
    db.initialize_table("typed_comments")
    db.initialize_table("mbti9k")
    db.vacuum_analyze()
    db.close()

@cli.command()
def summarize():
    """
    Create all summary tables/views from script.
    """
    db = pg.Connector()
    script = "schema/summary_tables.sql"
    db.execute_script(script)
    db.connection.commit()
    db.close()


if __name__ == "__main__":
    cli()
