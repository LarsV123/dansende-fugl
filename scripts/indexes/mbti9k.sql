CREATE INDEX IF NOT EXISTS mbti9k_author ON mbti9k_comments(author);
CREATE INDEX IF NOT EXISTS mbti9k_comment ON mbti9k_comments(comment);
CREATE INDEX IF NOT EXISTS mbti9k_type ON mbti9k_comments(type);
CREATE INDEX IF NOT EXISTS mbti9k_comments_num ON mbti9k_comments(comments_num);
CREATE INDEX IF NOT EXISTS mbti9k_author_comments ON mbti9k_comments(author, comment);
CREATE INDEX IF NOT EXISTS mbti9k_type_authors ON mbti9k_comments(type, author);