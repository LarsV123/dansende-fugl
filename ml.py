import sqlite3
import time
import nltk
from nltk import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import tensorflow as tf
import re
from keras.models import Sequential
from spellchecker import SpellChecker
from sklearn.model_selection import train_test_split
from models.nn_full import NNFull
from models.nn_individual import NNIndividual

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


class BatchTracker:
    """A simple batch tracker to feed data in batches to a function."""

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.count = 0
        self.complete = False

    def get_next_batch(self):
        """Provides the next batch of data or an empty list if the end of the data has been reached."""
        if self.complete:
            return []
        start = self.count
        end = start + self.batch_size
        if end >= len(self.data):
            end = len(self.data)
            self.complete = True
        self.count += self.batch_size
        return self.data[start:end]


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
    x_batcher = BatchTracker(data=x, batch_size=batch_size)
    y_batcher = BatchTracker(data=x, batch_size=batch_size)
    x_batch = x_batcher.get_next_batch()
    y_batch = y_batcher.get_next_batch()
    while len(x_batch):
        train_x = np.asarray(tkz.texts_to_matrix(x_batch)).astype(np.float32)
        model.fit(train_x, y_batch, callbacks=tb)
    return model


def evaluate_in_batch(model, tkz, x, y, batch_size):
    x_batcher = BatchTracker(data=x, batch_size=batch_size)
    y_batcher = BatchTracker(data=x, batch_size=batch_size)
    x_batch = x_batcher.get_next_batch()
    y_batch = y_batcher.get_next_batch()
    while len(x_batch):
        test_x = np.asarray(tkz.texts_to_matrix(x_batch)).astype(np.float32)
        model.evaluate(test_x, y_batch)
        x_batch = x_batcher.get_next_batch()
        y_batch = y_batcher.get_next_batch()
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


def convert_to_model_input(x, y, func, tokenizer):
    y = np.asarray([eval(func)(mbti_type) for mbti_type in y])
    x = np.asarray(tokenizer.texts_to_matrix(x, mode="tfidf")).astype(np.float32)
    return x, y


def train_model(x, y, tokenizer, model, verbose: bool = False):
    func = "get_individual_class"
    if isinstance(model, NNFull):
        func = "get_one_hot"
    x, y = convert_to_model_input(x, y, func, tokenizer)
    model.train(x=x, y=y, verbose=verbose)


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
    START_TIME = time.time()

    COMMENTS, TYPES = get_processed_data(size=10000, preprocess=False, folder="test")

    TOKENIZER = tf.keras.preprocessing.text.Tokenizer(1000)
    COMMENT_BATCHER = BatchTracker(data=COMMENTS, batch_size=int(len(COMMENTS) / 10))
    BATCH = COMMENT_BATCHER.get_next_batch()
    COUNT = 0
    while len(BATCH):
        print(f"Tokenizer fitting batch on {COUNT}")
        COUNT += 1
        TOKENIZER.fit_on_texts(BATCH)
        BATCH = COMMENT_BATCHER.get_next_batch()

    save = False
    # FULL_MODEL = NNFull(save=save)
    INDIVIDUAL_MODELS = NNIndividual(save=save)

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        COMMENTS, TYPES, test_size=0.25, random_state=1, stratify=None
    )
    # train_model(x_train, y_train, TOKENIZER, model=FULL_MODEL)
    train_model(X_TRAIN, Y_TRAIN, TOKENIZER, model=INDIVIDUAL_MODELS)

    print("--- %s seconds ---" % (time.time() - START_TIME))
