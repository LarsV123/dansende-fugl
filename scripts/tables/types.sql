-- Create aggregation table for personality types
CREATE TABLE IF NOT EXISTS types AS 
SELECT 
	type, 
	total_posts, 
	controversial_posts, 
	ROUND(controversial_posts*1.0 / total_posts, 5) AS controversiality_ratio
FROM (
	SELECT 
		type,
		COUNT(1) FILTER (WHERE controversiality > 0) AS controversial_posts,
		COUNT(1) AS total_posts
	FROM typed_comments
	WHERE author IS NOT NULL
	GROUP BY type
);