import pg


def get_user_data(db: pg.Connector, table: str, user: str):
    """
    Get all data for a specific user from a given table.
    """
    query = f"""
    --sql
    SELECT *
    FROM {table}
    WHERE author='{user}'
    ;
    """

    db.cursor.execute(query)
    rows = db.cursor.fetchall()
    db.connection.commit()
    return rows


def get_user_comments(db: pg.Connector, table: str, user: str):
    """
    Get only comments for a specific user from a given table.
    """
    query = f"""
    --sql
    SELECT comment
    FROM {table}
    WHERE author='{user}'
    ;
    """

    db.cursor.execute(query)
    rows = db.cursor.fetchall()
    db.connection.commit()
    return rows

if __name__ == "__main__":
    # Example usage
    db = pg.Connector()

    # Include metadata
    rows = get_user_data(db, "comments", "Famraine")
    for row in rows:
        print(row)

    # Only comments
    rows = get_user_comments(db, "comments", "Famraine")
    for row in rows:
        print(row)

    db.connection.close()
