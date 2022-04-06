-- Comments table including the 4 axes as new columns
DROP VIEW IF EXISTS comments_ax;
CREATE VIEW comments_ax AS (
  SELECT 
  *,
  SUBSTRING(mbti,1,1) AS ie,
  SUBSTRING(mbti,2,1) AS sn,
  SUBSTRING(mbti,3,1) AS tf,
  SUBSTRING(mbti,4,1) AS jp
  FROM 
  unique_comments
  WHERE NOT is_mbti_related
);
-- Comments per hour
CREATE TEMP TABLE total_comments_per_mbti AS (
  SELECT mbti,
    COUNT(*) AS total_comments
  FROM unique_comments
  GROUP BY mbti
);
CREATE TEMP TABLE temp_comments_per_hour AS (
  SELECT mbti,
    EXTRACT(
      HOUR
      FROM to_timestamp(created_utc)
    ) AS hour,
    COUNT(*) AS comments_per_hour
  FROM unique_comments
  GROUP BY mbti,
    hour
);
DROP TABLE IF EXISTS comments_per_hour;
CREATE TABLE comments_per_hour AS (
  SELECT mbti,
    comments_per_hour,
    total_comments,
    hour,
    comments_per_hour * 1.0 / total_comments * 1.0 AS share,
    SUBSTRING(mbti,1,1) AS ie,
    SUBSTRING(mbti,2,1) AS sn,
    SUBSTRING(mbti,3,1) AS tf,
    SUBSTRING(mbti,4,1) AS jp
  FROM total_comments_per_mbti AS a
    NATURAL JOIN temp_comments_per_hour
);
