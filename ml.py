import sqlite3

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from spellchecker import SpellChecker

nltk.download('punkt')
nltk.download('stopwords')

database = "data.db"
mbti_types = ["istj", "istp", "isfj", "isfp", "intj", "intp", "infj", "infp",
              "estj", "estp", "esfj", "esfp", "entj", "entp", "enfj", "enfp"]


def get_one_hot(mbti_type):
    idx = mbti_types.index(mbti_type)
    one_hot = np.zeros(len(mbti_types))
    one_hot[idx] = 1
    return one_hot


def get_typed_comments():
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
    while count < 1000:
        rows = cursor.fetchmany(1000)
        print(round(count / (22 * 10 ** 6), 2))
        count += 1000
        if len(rows) == 0:
            break
        for type, comment in rows:
            types.append(get_one_hot(type))
            comments.append(comment)
    return pd.DataFrame(list(zip(types, comments)), columns=["type", "comment"])


def get_matrix_from_text(tkz, x):
    batch_size = 10000
    matrix = tkz.texts_to_matrix(x[0:batch_size])
    i = batch_size
    while True:
        try:
            temp = tkz.texts_to_matrix(x[i:i + batch_size])
            matrix.append(temp)
            i += batch_size
        except:
            break
    return matrix


def remove_stop_words(text):
    result = []
    en_stopwords = stopwords.words('english')
    for token in text:
        if token not in en_stopwords:
            result.append(token)
    return result


# tokenizer = tf.keras.preprocessing.text.Tokenizer()
# df = get_typed_comments()
# tokenizer.fit_on_texts(df["comment"])
# train_x = get_matrix_from_text(tokenizer, df["comment"])
# train_y = df["type"][0:80000]
# test_x = get_matrix_from_text(tokenizer, df["comment"])
# test_y = df["type"][80000]

print("Loading data...")
df = get_typed_comments()
print("Complete!")
# Create word tokens from each comment
print("Creating tokens...")
df["comment"] = df["comment"].apply(lambda x: nltk.word_tokenize(x))
print("Complete!")
# Remove stopwords
print("Removing stopwords...")
df["comment"] = df["comment"].apply(lambda text: remove_stop_words(text))
print("Complete!")
# Apply spell checking
print("Applying spellcheck...")
spell = SpellChecker()
df["comment"] = df["comment"].apply(lambda text: [spell.correction(word) for word in text])
print("Complete!")
# Stemming
print("Applying stemming...")
porter = PorterStemmer()
df["comment"] = df["comment"].apply(lambda text: [porter.stem(word) for word in text])
print("Complete!")
# model = Sequential()
# model.add(Dense(32, input_dim=16, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(train_x, train_y, batch_size=8)
# model.evaluate(test_x, test_y)
