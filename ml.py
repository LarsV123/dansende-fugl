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
from sklearn.model_selection import train_test_split

nltk.download("punkt")
nltk.download("stopwords")

database = "data.db"
MBTI_TYPES = [
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


def get_one_hot(mbti_type: str):
    """:param mbti_type: One of the sixteen mbti personality types.
    :returns: A one hot encoding of the given mbti."""
    idx = MBTI_TYPES.index(mbti_type.lower())
    one_hot = [0 for _ in range(len(MBTI_TYPES))]
    one_hot[idx] = 1
    return one_hot


def get_individual_class(mbti_type):
    """:param mbti_type: One of the sixteen mbti personality types.
    :returns A one hot encoding of for each of the four individual axis of mbti: I/E, S/N, F/T, J/P."""
    individual_class = {"i": 0, "e": 1, "s": 0, "n": 1, "f": 0, "t": 1, "j": 0, "p": 1}
    classification = []
    for char in mbti_type:
        classification.append(individual_class[char])
    return classification


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


def remove_stop_words(text):
    en_stopwords = stopwords.words("english")
    return [token for token in text if token not in en_stopwords]


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
        model.fit(train_x, y[start:end], callbacks=tb)
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


def pre_process_batch(arr, batch_size, folder):
    count = 0
    idx = 0
    run = True
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
        idx += 1
        count += batch_size


def train_model(comments, types, tokenizer, full_type: bool):
    x_train, x_test, y_train, y_test = train_test_split(
        comments, types, test_size=0.25, random_state=1, stratify=types
    )
    func = "get_individual_class"
    if full_type:
        func = "get_one_hot"
    y_train = np.asarray([eval(func)(mbti_type) for mbti_type in y_train])
    y_test = np.asarray([eval(func)(mbti_type) for mbti_type in y_test])
    model_input = np.asarray(tokenizer.texts_to_matrix(x_train, mode="tfidf")).astype(
        np.float32
    )
    if full_type:
        models = full_model()
        file_path = f"./models/full_model_{comments.size}"
        models.fit(
            model_input,
            y_train,
            batch_size=32,
            epochs=10,
            validation_split=0.2,
        )
        models.save(file_path)
    else:
        models = individual_models()
        dimensions = ["I_E", "S_N", "F_T", "J_P"]
        file_path = f"./models/individual_model_{comments.size}"
        for _ in range(len(models)):
            print(f"Training on dimension: {dimensions[_]}")
            model = models[_]
            print(y_train[0:5, _])
            model.fit(
                model_input,
                y_train[:, _],
                batch_size=32,
                epochs=10,
                validation_split=0.2,
            )
            model.save(file_path + dimensions[_])
    return models


def individual_models():
    """Build and return four individual models, meant to predict one of the four axis in mbti each."""
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
    return models


def full_model():
    """Build and return a model meant for predicting one out of 16 personality types."""
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
    return model


def get_processed_data(size, preprocess, folder="processed"):
    print("Loading data...")
    if preprocess:
        types, comments = get_typed_comments(batch_size=int(size / 10), n=size)
        pre_process_batch(comments, int(len(comments) / 10), folder)
        np.savez_compressed(f"./data/{folder}/types.npz", np.asarray(types))
    comments = []
    i = 0
    while True:
        try:
            temp = np.load(f"./data/{folder}/comments_{i}.npz", allow_pickle=True)[
                "arr_0"
            ]
            comments.append(temp)
            print(f"Comment batch {i} loaded successfully!")
            i += 1
        except FileNotFoundError:
            break
    comments = np.hstack(np.asarray(comments))
    types = np.load(f"./data/{folder}/types.npz", allow_pickle=True)["arr_0"]
    print("Done!")
    return comments, types


if __name__ == "__main__":
    start_time = time.time()

    COMMENTS, TYPES = get_processed_data(size=100, preprocess=False)

    TOKENIZER = tf.keras.preprocessing.text.Tokenizer(10000)
    TOKENIZER.fit_on_texts(COMMENTS)

    MODELS = train_model(COMMENTS, TYPES, TOKENIZER, full_type=False)
    INDIVIDUAL_MODELS = train_model(COMMENTS, TYPES, TOKENIZER, full_type=True)

    print("--- %s seconds ---" % (time.time() - start_time))
