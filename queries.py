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

def query3(db: pg.Connector):
    """
    Group comments by hour (24 buckets) and MBTI type.
    """
    query = """
    --sql
    SELECT mbti, EXTRACT(HOUR FROM to_timestamp(created_utc)) AS hour, COUNT(*) AS comments_per_hour
    FROM comments
    GROUP BY mbti, hour
    ;
    """
    db.cursor.execute(query)
    rows = db.cursor.fetchall()
    print(tabulate(rows))

def query4(db: pg.Connector):
    query = """
    --sql
    SELECT mbti_type as mbti, subreddit, COUNT(*) AS comment_count
    FROM typed_comments
    GROUP BY subreddit, mbti
    ORDER BY comment_count DESC
    LIMIT 20
    ;
    """



    query = """
    --sql
    SELECT comments.subreddit, comments.mbti
    FROM comments
    INNER JOIN (
        SELECT subreddit, COUNT(*) as total_comments
        FROM comments
        GROUP BY subreddit
        ORDER BY total_comments DESC
        LIMIT 20
    ) AS top_subreddits
    ON comments.subreddit = top_subreddits.subreddit
    ;
    """
    db.cursor.execute(query)
    rows = db.cursor.fetchall()
    print(tabulate(rows))


def query6(db: pg.Connector):
    """
    How many users of each type are in the dataset in total. Number and percentage.
    """
    query = f"""
    --sql
    CREATE MATERIALIZED VIEW IF NOT EXISTS percentage_comments_per_mbti AS
    SELECT mbti_type, 
        COUNT(*) AS total_comments, 
        COUNT(*) * 1.0/ (
            SELECT COUNT(*) * 1.0 FROM typed_comments
            ) AS percentage_comments
    FROM typed_comments
    GROUP BY mbti_type
    ;
    """
    query = f"""
    --sql
    CREATE MATERIALIZED VIEW comments_per_mbti2 AS
    SELECT mbti, COUNT(*) AS mbti_comment_count
    FROM comments
    GROUP BY mbti
    ;
    """
    db.cursor.execute(query)
    db.connection.commit()
    # rows = db.cursor.fetchall()
    # headers = [desc[0] for desc in db.cursor.description]
    # print(tabulate(rows, headers=headers))


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
    # query2(db)
    # query4(db)
    query3(db)

    db.connection.close()
