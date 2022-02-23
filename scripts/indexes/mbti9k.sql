CREATE INDEX IF NOT EXISTS mbti9k_author ON mbti9k_comments(author);
CREATE INDEX IF NOT EXISTS mbti9k_type ON mbti9k_comments(type);
CREATE INDEX IF NOT EXISTS mbti9k_comments_num ON mbti9k_comments(comments_num);