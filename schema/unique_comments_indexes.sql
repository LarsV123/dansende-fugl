-- Single column indexes
CREATE INDEX IF NOT EXISTS unique_comments_created_utc ON unique_comments(created_utc);
CREATE INDEX IF NOT EXISTS unique_comments_author ON unique_comments(author);
CREATE INDEX IF NOT EXISTS unique_comments_mbti ON unique_comments(mbti);
CREATE INDEX IF NOT EXISTS unique_comments_subreddit ON unique_comments(subreddit);
CREATE INDEX IF NOT EXISTS unique_comments_score ON unique_comments(score);
CREATE INDEX IF NOT EXISTS unique_comments_gilded ON unique_comments(gilded);
CREATE INDEX IF NOT EXISTS unique_comments_word_count ON unique_comments(word_count);
CREATE INDEX IF NOT EXISTS unique_comments_word_count_quoteless ON unique_comments(word_count_quoteless);
CREATE INDEX IF NOT EXISTS unique_comments_quote_to_text_ratio ON unique_comments(quote_to_text_ratio);
CREATE INDEX IF NOT EXISTS unique_comments_is_mbti_related ON unique_comments(is_mbti_related);
CREATE INDEX IF NOT EXISTS unique_comments_lang ON unique_comments(lang);
-- Compound indexes
CREATE INDEX IF NOT EXISTS unique_comments_type_to_subreddit ON unique_comments(mbti, subreddit);
CREATE INDEX IF NOT EXISTS unique_comments_type_to_author ON unique_comments(mbti, author);
CREATE INDEX IF NOT EXISTS unique_comments_controversiality_to_type ON unique_comments(controversiality, mbti);