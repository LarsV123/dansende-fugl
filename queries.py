import pg
from tabulate import tabulate

from utils import time_this

@time_this
def query1_1(db: pg.Connector):
    """
    Find average post score per personality type.
    """
    query = """
    --sql
    SELECT mbti, ROUND(AVG(score), 2) AS avg_score, COUNT(*) AS num_comments
    FROM posts
    GROUP BY mbti
    ORDER BY avg_score DESC
    ;
    """
    db.cursor.execute(query)
    rows = db.cursor.fetchall()
    print("Average post score per personality type.")
    headers = ["MBTI", "Avg. score", "Post count"]
    print(tabulate(rows, headers=headers, tablefmt="pretty"))

@time_this
def query1_2(db: pg.Connector):
    """
    Find average comment score per personality type.
    """
    query = """
    --sql
    SELECT mbti, ROUND(AVG(score)) AS avg_score, COUNT(*) AS num_comments
    FROM comments
    GROUP BY mbti
    ORDER BY avg_score DESC
    ;
    """
    db.cursor.execute(query)
    rows = db.cursor.fetchall()
    print("Average comment score per personality type.")
    headers = ["MBTI", "Avg. score", "Post count"]
    print(tabulate(rows, headers=headers, tablefmt="pretty"))

@time_this
def query2(db: pg.Connector):
    """
    Find percentage of controversial comments per personality type.
    """
    query = """
    --sql
    SELECT mbti, 
        controversial_comments, 
        total_comments, 
        ROUND(controversial_comments*1.0 / total_comments, 5) * 100 AS ratio
    FROM (SELECT 
		mbti,
		COUNT(*) FILTER (WHERE controversiality > 0) AS controversial_comments,
		COUNT(*) AS total_comments
        FROM comments
        GROUP BY mbti
    ) AS a
    UNION
    SELECT 
        'ALL', 
        controversial_comments, 
        total_comments, 
        ROUND(controversial_comments*1.0 / total_comments, 5) * 100 AS ratio 
    FROM (
		SELECT
            COUNT(*) FILTER (WHERE controversiality > 0) AS controversial_comments,
            COUNT(*) AS total_comments
        FROM comments
        ) AS c
    ORDER BY total_comments DESC
    ;
    """
    db.cursor.execute(query)
    rows = db.cursor.fetchall()
    print("Average controversiality of comments per personality type.")
    headers = ["MBTI", "Controversial comments", "Total comments", "Percentage controversial comments"]
    print(tabulate(rows, headers=headers, tablefmt="pretty"))



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

    # # Include metadata
    # rows = get_user_data(db, "comments", "Famraine")
    # for row in rows:
    #     print(row)

    # # Only comments
    # rows = get_user_comments(db, "comments", "Famraine")
    # for row in rows:
    #     print(row)

    # query1_1(db)
    # query1_2(db)
    query2(db)

    db.connection.close()
