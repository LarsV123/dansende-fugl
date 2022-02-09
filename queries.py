import sqlite3


def get_user_data(connection: sqlite3.Connection, table: str, user: str):
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

    cursor = connection.cursor()
    rows = cursor.execute(query).fetchall()
    return rows


def get_user_comments(connection: sqlite3.Connection, table: str, user: str):
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

    cursor = connection.cursor()
    rows = cursor.execute(query).fetchall()
    return rows


if __name__ == "__main__":
    # Example usage
    database = "data.db"
    connection = sqlite3.connect(database)

    # Include metadata
    rows = get_user_data(connection, "mbti9k_comments", "Famraine")
    for row in rows:
        print(row)

    # Only comments
    rows = get_user_comments(connection, "mbti9k_comments", "Famraine")
    for row in rows:
        print(row)
