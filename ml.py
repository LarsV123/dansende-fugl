import sqlite3
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import tensorflow as tf
import pandas as pd
import re
from keras.layers import Dense
from keras.models import Sequential
from spellchecker import SpellChecker

nltk.download("punkt")
nltk.download("stopwords")

database = "data.db"
mbti_types = [
    "istj",
    "istp",
    "isfj",
    "isfp",
    "intj",
    "intp",
    "infj",
    "infp",
    "estj",
    "estp",
    "esfj",
    "esfp",
    "entj",
    "entp",
    "enfj",
    "enfp",
]


def get_one_hot(mbti_type):
    idx = mbti_types.index(mbti_type)
    one_hot = [0 for _ in range(len(mbti_types))]
    one_hot[idx] = 1
    return one_hot


def get_typed_comments(size):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    query = """
    --sql
    SELECT type, comment
    FROM typed_comments
    ;
    """
    cursor.execute(query)
    comments = []
    types = []
    count = 0
    while count < size:
        rows = cursor.fetchmany(1000)
        print(round(count / size, 3))
        count += 1000
        if len(rows) == 0:
            break
        for type, comment in rows:
            types.append(get_one_hot(type))
            comments.append(comment)
    return pd.DataFrame(
        list(zip(np.array(types), comments)), columns=["type", "comment"]
    )


def get_matrix_from_text(tkz, x):
    batch_size = 1000
    matrix = tkz.texts_to_matrix(x[0:batch_size])
    i = batch_size
    while True:
        try:
            temp = tkz.texts_to_matrix(x[i : i + batch_size])
            matrix = np.concatenate([matrix, temp])
            i += batch_size
        except IndexError:
            break
    return matrix


def remove_stop_words(text):
    result = []
    en_stopwords = stopwords.words("english")
    for token in text:
        if token not in en_stopwords:
            result.append(token)
    return result


start_time = time.time()
print("Loading data...")
size = 100000
df = get_typed_comments(size)
print("Complete!")


# Create word tokens from each comment
print("Creating tokens...")
df["comment"] = df["comment"].apply(lambda x: nltk.word_tokenize(x))
print("Complete!")
# Remove stopwords
print("Removing stopwords...")
df["comment"] = df["comment"].apply(lambda text: remove_stop_words(text))
print("Complete!")
# Remove special characters
print("Removing special characters...")
df["comment"] = df["comment"].apply(
    lambda text: [re.sub("[^a-zA-Z0-9]+", "", _) for _ in text]
)
# Apply spell checking (Removed for now, way too slow)
# print("Applying spellcheck...")
# spell = SpellChecker()
# df["comment"] = df["comment"].apply(lambda text: [spell.correction(word) for word in text])
# print("Complete!")
# Stemming
# print("Applying stemming...")
# porter = PorterStemmer()
# df["comment"] = df["comment"].apply(lambda text: [porter.stem(word) for word in text])
# print("Complete!")

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df["comment"])
train_x = np.asarray(
    get_matrix_from_text(tokenizer, df["comment"][: int(0.8 * size)])
).astype(np.float32)
train_y = np.stack(df["type"][0 : int(0.8 * size)])
test_x = np.asarray(
    get_matrix_from_text(tokenizer, df["comment"][int(0.8 * size) :])
).astype(np.float32)
test_y = np.stack(df["type"][int(0.8 * size) :])
model = Sequential()
model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="softmax"))
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]
)
model.fit(train_x, train_y)
model.evaluate(test_x, test_y)
print("--- %s seconds ---" % (time.time() - start_time))
