import sqlite3
import time
import nltk
from nltk import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import tensorflow as tf
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


def get_typed_comments(batch_size, n):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    query = """
    --sql
    SELECT type, comment
    FROM mbti9k_comments
    ;
    """
    cursor.execute(query)
    comments = []
    types = []
    count = 0
    while count < n:
        rows = cursor.fetchmany(batch_size)
        print(f"{round(count * 100 / n, 2)} %")
        count += batch_size
        if len(rows) == 0:
            break
        for type, comment in rows:
            types.append(type)
            comments.append(comment)
    return types, comments


def get_matrix_from_text(tkz, x, batch_size):
    matrix = tkz.texts_to_matrix(x[0:batch_size])
    i = batch_size
    run = True
    while run:
        start = i
        end = i + batch_size
        if end >= len(x):
            end = len(x) - i
            run = False
        temp = tkz.texts_to_matrix(x[i : i + batch_size])
        matrix = np.concatenate([matrix, temp])
        print(f"Tokenize: {i}")
        i += batch_size
    return matrix


def remove_stop_words(text):
    result = []
    en_stopwords = stopwords.words("english")
    for token in text:
        if token not in en_stopwords:
            result.append(token)
    return result


def pre_process(arr):
    print("Creating tokens...")
    toktok = ToktokTokenizer()
    arr = [toktok.tokenize(x) for x in arr]
    print("Done!")

    print("Removing stopwords...")
    arr = [remove_stop_words(x) for x in arr]
    print("Done!")

    print("Removing special characters...")
    arr = [[re.sub("[^a-zA-Z0-9]+", "", x) for x in y] for y in arr]
    arr = [[x for x in y if x != ""] for y in arr]
    print("Done!")

    # Very slow, removed for now
    # print("Applying spellcheck...")
    # spell = SpellChecker()
    # arr = [[spell.correction(x) for x in y] for y in arr]
    # print("Done!")

    # Very slow, removed for now
    # print("Applying stemming...")
    # porter = PorterStemmer()
    # arr = [[porter.stem(x) for x in y] for y in arr]
    # print("Done!")
    return arr


def train_in_batch(model, tkz, x, y, batch_size, tb=None):
    count = 0
    run = False
    while run:
        start = count
        end = count + batch_size
        if end >= len(x):
            end = len(x)
            run = False
        train_x = np.asarray(tkz.texts_to_matrix(x[start:end])).astype(np.float32)
        if tb:
            model.fit(train_x, y[start:end], callbacks=tb)
        else:
            model.fit(train_x, y[start:end])
        count += batch_size
    return model


def evaluate_in_batch(model, tkz, x, y, batch_size):
    count = 0
    run = True
    print("Evaluation on test: ")
    while run:
        start = count
        end = count + batch_size
        if end >= len(x):
            end = len(x)
            run = False
        test_x = np.asarray(tkz.texts_to_matrix(x[start:end])).astype(np.float32)
        model.evaluate(test_x, y[start:end])
        count += batch_size
    return model


if __name__ == "__main__":
    start_time = time.time()
    size = 1000
    load_from_file = False
    print("Loading data...")
    if load_from_file:
        # Load from compressed file
        comments = np.load("processed_comments.npz", allow_pickle=True)["arr_0"]
        types = np.load("types.npz", allow_pickle=True)["arr_0"]
    else:
        types, comments = get_typed_comments(batch_size=int(size / 10), n=size)
        comments = pre_process(comments)
        # Save to compressed file
        np.savez_compressed("processed_comments.npz", np.asarray(comments))
        np.savez_compressed("types.npz", np.asarray(types))
    print("Done!")

    train_comments = np.asarray(comments[: int(0.8 * size)])
    train_types = np.asarray(
        [get_one_hot(author) for author in types[0 : int(0.8 * size)]]
    )
    test_comments = np.asarray(comments[int(0.8 * size) :])
    test_types = np.asarray(
        [get_one_hot(author) for author in types[int(0.8 * size) :]]
    )
    tokenizer = tf.keras.preprocessing.text.Tokenizer(10000)
    tokenizer.fit_on_texts(comments)

    MODEL = Sequential()
    MODEL.add(Dense(32, activation="relu"))
    MODEL.add(Dense(32, activation="relu"))
    MODEL.add(Dense(32, activation="relu"))
    MODEL.add(Dense(32, activation="relu"))
    MODEL.add(Dense(32, activation="relu"))
    MODEL.add(Dense(16, activation="softmax"))
    MODEL.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["categorical_accuracy"],
    )
    batch = 100
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    # MODEL = train_in_batch(MODEL, tokenizer, train_comments, train_types, batch, tensorboard_callback)
    # MODEL = evaluate_in_batch(MODEL, tokenizer, test_comments, test_types, batch)
    MODEL.fit(
        np.asarray(tokenizer.texts_to_matrix(train_comments)).astype(np.float32),
        train_types,
        batch_size=8,
        epochs=5,
    )
    MODEL.evaluate(
        np.asarray(tokenizer.texts_to_matrix(test_comments)).astype(np.float32),
        test_types,
    )
    print("--- %s seconds ---" % (time.time() - start_time))
