-- Single column indexes
CREATE INDEX IF NOT EXISTS mbti9k_author ON mbti9k(author);
CREATE INDEX IF NOT EXISTS mbti9k_mbti_type ON mbti9k(mbti_type);
CREATE INDEX IF NOT EXISTS mbti9k_subreddits_commented ON mbti9k(subreddits_commented);
CREATE INDEX IF NOT EXISTS mbti9k_mbti_subreddits_commented ON mbti9k(mbti_subreddits_commented);
CREATE INDEX IF NOT EXISTS mbti9k_wc ON mbti9k(wc);
CREATE INDEX IF NOT EXISTS mbti9k_comments_num ON mbti9k(comments_num);
-- Compound indexes
CREATE INDEX IF NOT EXISTS mbti9k_type_to_author ON mbti9k(mbti_type, author);