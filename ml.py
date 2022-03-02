import sqlite3
import time
import nltk
from nltk import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import tensorflow as tf
import re
from keras.layers import Dense, Dropout
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
    en_stopwords = stopwords.words("english")
    return [token for token in text if token not in en_stopwords]


def pre_process_batch(arr, batch_size, folder):
    count = 0
    idx = 0
    run = True
    comments = []
    while run:
        print(f"Processing batch {idx}")
        start = count
        end = count + batch_size
        if end >= len(arr):
            end = len(arr)
            run = False
        curr_arr = arr[start:end]
        print("Creating tokens...")
        toktok = ToktokTokenizer()
        curr_arr = [toktok.tokenize(x) for x in curr_arr]

        print("Removing stopwords...")
        curr_arr = [remove_stop_words(x) for x in curr_arr]

        print("Removing special characters...")
        curr_arr = [[re.sub("[^a-zA-Z0-9!?]+", "", x) for x in y] for y in curr_arr]
        curr_arr = [[x for x in y if x != ""] for y in curr_arr]

        # Very slow, removed for now
        # print("Applying spellcheck...")
        # spell = SpellChecker()
        # arr = [[spell.correction(x) for x in y] for y in curr_arr]

        # Also very slow, removed for now
        # print("Applying stemming...")
        # porter = PorterStemmer()
        # arr = [[porter.stem(x) for x in y] for y in curr_arr]

        # Save to compressed file
        np.savez_compressed(f"./data/{folder}/comments_{idx}", np.asarray(curr_arr))
        comments.append(curr_arr)
        idx += 1
        count += batch_size


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


def get_individual_class(mbti_type):
    individual_class = {"i": 0, "e": 1, "s": 0, "n": 1, "f": 0, "t": 1, "j": 0, "p": 1}
    classification = []
    for char in mbti_type:
        classification.append(individual_class[char])
    return classification


def train_individual_axis(comments, types, tokenizer):
    train_idx = [_ for _ in range(int(0.8 * size))]
    train_comments = np.take(comments, train_idx)
    train_types = np.asarray(
        [get_individual_class(author) for author in types[0 : int(0.8 * size)]]
    )
    test_idx = [_ for _ in range(int(0.8 * size), size)]
    test_comments = np.take(comments, test_idx)
    test_types = np.asarray(
        [get_individual_class(author) for author in types[int(0.8 * size) :]]
    )
    models = []
    for _ in range(4):
        model = Sequential()
        for i in range(4):
            model.add(Dense(128, activation="relu", kernel_regularizer="L2"))
            model.add(Dropout(0.4))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        models.append(model)
    dimensions = ["I/E", "S/N", "F/T", "J/P"]
    model_input = np.asarray(
        tokenizer.texts_to_matrix(train_comments, mode="tfidf")
    ).astype(np.float32)
    for _ in range(len(models)):
        print(f"Training on dimension: {dimensions[_]}")
        models[_].fit(
            model_input,
            train_types[:, _],
            batch_size=32,
            epochs=10,
            validation_split=0.2,
        )
    predict(
        models,
        np.asarray(tokenizer.texts_to_matrix(test_comments[0], mode="tfidf")).astype(
            np.float32
        ),
        test_types[0],
    )
    return models


def predict(models, x, y):
    pred_type = y
    for model in models:
        temp = model(x)
    return


def train_full_type(comments, types, tokenizer):
    train_idx = [_ for _ in range(int(0.8 * size))]
    train_comments = np.take(comments, train_idx)
    train_types = np.asarray(
        [get_one_hot(author) for author in types[0 : int(0.8 * size)]]
    )
    test_idx = [_ for _ in range(int(0.8 * size), size)]
    test_comments = np.take(comments, test_idx)
    test_types = np.asarray(
        [get_one_hot(author) for author in types[int(0.8 * size) :]]
    )
    model = Sequential()
    for i in range(4):
        model.add(Dense(64, activation="relu", kernel_regularizer="L2"))
        model.add(Dropout(0.2))
    model.add(Dense(16, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["categorical_accuracy"],
    )
    batch = 100
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    # MODEL = train_in_batch(MODEL, tokenizer, train_comments, train_types, batch, tensorboard_callback)
    # MODEL = evaluate_in_batch(MODEL, tokenizer, test_comments, test_types, batch)
    model.fit(
        np.asarray(tokenizer.texts_to_matrix(train_comments, mode="tfidf")).astype(
            np.float32
        ),
        train_types,
        batch_size=32,
        epochs=25,
        validation_split=0.2,
        callbacks=tensorboard_callback,
    )
    # print("--- Evaluate ---")
    # MODEL.evaluate(
    #     np.asarray(tokenizer.texts_to_matrix(test_comments)).astype(np.float32),
    #     test_types,
    # )
    return model


if __name__ == "__main__":
    start_time = time.time()
    size = 500
    folder = "test"
    preprocess_data = False
    print("Loading data...")
    if preprocess_data:
        TYPES, COMMENTS = get_typed_comments(batch_size=int(size / 10), n=size)
        pre_process_batch(COMMENTS, int(len(COMMENTS) / 10), folder)
        np.savez_compressed(f"./data/{folder}/types.npz", np.asarray(TYPES))
    COMMENTS = []
    i = 0
    while True:
        try:
            temp = np.load(f"./data/{folder}/comments_{i}.npz", allow_pickle=True)[
                "arr_0"
            ]
            COMMENTS.append(temp)
            print(f"Comment batch {i} loaded successfully!")
            i += 1
        except FileNotFoundError:
            break
    COMMENTS = np.hstack(np.asarray(COMMENTS))
    TYPES = np.load(f"./data/{folder}/types.npz", allow_pickle=True)["arr_0"]
    print("Done!")

    TOKENIZER = tf.keras.preprocessing.text.Tokenizer(10000)
    TOKENIZER.fit_on_texts(COMMENTS)

    MODELS = train_individual_axis(COMMENTS, TYPES, TOKENIZER)

    print("--- %s seconds ---" % (time.time() - start_time))
