DROP TABLE IF EXISTS unique_comments;
CREATE TABLE unique_comments AS
SELECT DISTINCT *
FROM comments;