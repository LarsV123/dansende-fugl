DROP TABLE IF EXISTS hour_confint;
CREATE TABLE hour_confint AS (
    SELECT mbti, 
        hour, 
        share,
        share - 1.96*SQRT(share*(1-share)/total_comments) AS lci,
        share + 1.96*SQRT(share*(1-share)/total_comments) AS uci,
        comments_per_hour,
        total_comments
    FROM comments_per_hour
);


DROP TABLE IF EXISTS hour_ie;
CREATE TEMP TABLE hour_ie AS (
    SELECT ie, 
        hour, 
        SUM(total_comments) AS n,
        SUM(comments_per_hour) AS n_h,
        1.0 * SUM(comments_per_hour) / SUM(total_comments) AS p
        FROM comments_per_hour 
        GROUP BY ie, hour
);
DROP TABLE IF EXISTS hour_confint_ie;
CREATE TABLE hour_confint_ie AS (
    SELECT ie, 
        hour, 
        p,
        n,
        p - 1.96*SQRT(p*(1-p)/n) AS lci,
        p + 1.96*SQRT(p*(1-p)/n) AS uci
    FROM hour_ie
);


DROP TABLE IF EXISTS hour_sn;
CREATE TEMP TABLE hour_sn AS (
    SELECT sn, 
        hour, 
        SUM(total_comments) AS n,
        SUM(comments_per_hour) AS n_h,
        1.0 * SUM(comments_per_hour) / SUM(total_comments) AS p
        FROM comments_per_hour 
        GROUP BY sn, hour
);
DROP TABLE IF EXISTS hour_confint_sn;
CREATE TABLE hour_confint_sn AS (
    SELECT sn, 
        hour, 
        p,
        n,
        p - 1.96*SQRT(p*(1-p)/n) AS lci,
        p + 1.96*SQRT(p*(1-p)/n) AS uci
    FROM hour_sn
);


DROP TABLE IF EXISTS hour_tf;
CREATE TEMP TABLE hour_tf AS (
    SELECT tf, 
        hour, 
        SUM(total_comments) AS n,
        SUM(comments_per_hour) AS n_h,
        1.0 * SUM(comments_per_hour) / SUM(total_comments) AS p
        FROM comments_per_hour 
        GROUP BY tf, hour
);
DROP TABLE IF EXISTS hour_confint_tf;
CREATE TABLE hour_confint_tf AS (
    SELECT tf, 
        hour, 
        p,
        n,
        p - 1.96*SQRT(p*(1-p)/n) AS lci,
        p + 1.96*SQRT(p*(1-p)/n) AS uci
    FROM hour_tf
);


DROP TABLE IF EXISTS hour_jp;
CREATE TEMP TABLE hour_jp AS (
    SELECT jp, 
        hour, 
        SUM(total_comments) AS n,
        SUM(comments_per_hour) AS n_h,
        1.0 * SUM(comments_per_hour) / SUM(total_comments) AS p
        FROM comments_per_hour 
        GROUP BY jp, hour
);
DROP TABLE IF EXISTS hour_confint_jp;
CREATE TABLE hour_confint_jp AS (
    SELECT jp, 
        hour, 
        p,
        n,
        p - 1.96*SQRT(p*(1-p)/n) AS lci,
        p + 1.96*SQRT(p*(1-p)/n) AS uci
    FROM hour_jp
);


DROP TABLE IF EXISTS hour_infj;
CREATE TEMP TABLE hour_infj AS (
    SELECT mbti, 
        hour, 
        SUM(total_comments) AS n,
        SUM(comments_per_hour) AS n_h,
        1.0 * SUM(comments_per_hour) / SUM(total_comments) AS p
        FROM comments_per_hour 
        WHERE mbti = 'infj'
        GROUP BY hour, mbti
);
DROP TABLE IF EXISTS hour_confint_infj;
CREATE TABLE hour_confint_infj AS (
    SELECT mbti, 
        hour, 
        p,
        n,
        p - 1.96*SQRT(p*(1-p)/n) AS lci,
        p + 1.96*SQRT(p*(1-p)/n) AS uci
    FROM hour_infj
);