import copy
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

from models.baseline import Baseline
from models.nn_full import NNFull
from models.nn_individual import NNIndividual
import matplotlib.pyplot as plt
from models.lgbm import LGBM
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

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

    def __init__(self, data_size, batch_size):
        self.data_size = data_size
        self.batch_size = batch_size
        self.count = 0
        self.complete = False

    def get_next_batch(self):
        """Provides the next batch of data or an empty list if the end of the data has been reached."""
        if self.complete:
            return []
        start = self.count
        end = start + self.batch_size
        if end >= self.data_size:
            end = self.data_size
            self.complete = True
        self.count += self.batch_size
        return [start, end]


def get_one_hot(mbti_type: str):
    """:param mbti_type: One of the sixteen mbti personality types.
    :returns: A one hot encoding of the given mbti."""
    idx = MBTI_TYPES.index(mbti_type.lower())
    one_hot = [0 for _ in range(len(MBTI_TYPES))]
    one_hot[idx] = 1
    return one_hot


def get_individual_one_hot(mbti_type):
    """:param mbti_type: One of the sixteen mbti personality types.
    :returns A one hot encoding of for each of the four individual axis of mbti: I/E, S/N, F/T, J/P."""
    individual_class = {"i": 0, "e": 1, "s": 0, "n": 1, "f": 0, "t": 1, "j": 0, "p": 1}
    classification = []
    for char in mbti_type:
        classification.append(individual_class[char])
    return classification


def get_type_from_individual_one_hot(one_hot_mbti_type, dim=None):
    if not dim:
        dim = [i for i in range(len(one_hot_mbti_type))]
    dim_map = [{0: "i", 1: "e"}, {0: "s", 1: "n"}, {0: "f", 1: "t"}, {0: "j", 1: "p"}]
    mbti_type = ""
    for i in range(len(dim)):
        mbti_type += dim_map[dim[i]][one_hot_mbti_type[i]]
    return mbti_type


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
    x_batcher = BatchTracker(data_size=len(x), batch_size=batch_size)
    y_batcher = BatchTracker(data_size=len(y), batch_size=batch_size)
    x_batch = x_batcher.get_next_batch()
    y_batch = y_batcher.get_next_batch()
    while len(x_batch):
        train_x = np.asarray(tkz.texts_to_matrix(x[x_batch[0]:x_batch[1]])).astype(np.float32)
        model.fit(train_x, y[y_batch[0]:y_batch[1]], callbacks=tb)
        x_batch = x_batcher.get_next_batch()
        y_batch = y_batcher.get_next_batch()
    return model


def evaluate_in_batch(model, tkz, x, y, batch_size):
    x_batcher = BatchTracker(data_size=len(x), batch_size=batch_size)
    y_batcher = BatchTracker(data_size=len(y), batch_size=batch_size)
    x_batch = x_batcher.get_next_batch()
    y_batch = y_batcher.get_next_batch()
    while len(x_batch):
        test_x = np.asarray(tkz.texts_to_matrix(x[x_batch[0]:x_batch[1]])).astype(np.float32)
        model.evaluate(test_x, y[y_batch[0]:y_batch[1]])
        x_batch = x_batcher.get_next_batch()
        y_batch = y_batcher.get_next_batch()
    return model


def pre_process_batch(arr, batch_size, folder, two_gram: bool):
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
        curr_arr = [toktok.tokenize(x.lower()) for x in curr_arr]

        print("Removing special characters...")
        curr_arr = [[re.sub("[^a-zA-Z!?]+", "", x) for x in y] for y in curr_arr]
        curr_arr = [[x for x in y if x != ""] for y in curr_arr]

        print("Removing stopwords...")
        curr_arr = [remove_stop_words(x) for x in curr_arr]

        if two_gram:
            print("Combining into two-grams...")
            curr_arr = [
                [f"{x[i]} {x[i + 1]}" for i in range(len(x) - 1)] for x in curr_arr
            ]

        # Very slow, removed for now
        # print("Applying spellcheck...")
        # spell = SpellChecker()
        # curr_arr = [[spell.correction(x) for x in y] for y in curr_arr]

        # Also very slow, removed for now
        # print("Applying stemming...")
        # porter = PorterStemmer()
        # curr_arr = [[porter.stem(x) for x in y] for y in curr_arr]

        # Save to compressed file
        np.savez_compressed(f"./data/{folder}/comments_{idx}", np.asarray(curr_arr))
        idx += 1
        count += batch_size


def rescale_along_row(x):
    return (x - x.min()) / x.max()


def convert_to_model_input(x, y, func, tokenizer):
    y = np.asarray([eval(func)(mbti_type) for mbti_type in y])
    x = np.asarray(tokenizer.texts_to_matrix(x, mode="tfidf")).astype(np.float32)
    # x = np.apply_along_axis(rescale_along_row, 1, x)
    return x, y


def train_model(x, y, tokenizer, model, dim=None, verbose: bool = False):
    func = "get_individual_one_hot"
    if isinstance(model, NNFull):
        func = "get_one_hot"
    if len(x) > 2000:
        x_batcher = BatchTracker(data_size=len(x), batch_size=2000)
        y_batcher = BatchTracker(data_size=len(y), batch_size=2000)
        x_batch = x_batcher.get_next_batch()
        y_batch = y_batcher.get_next_batch()
        x_input = []
        y_input = []
        while len(x_batch):
            x_batch, y_batch = convert_to_model_input(x[x_batch[0]:x_batch[1]], y[y_batch[0]:y_batch[1]], func,
                                                      tokenizer)
            x_input.append(x_batch)
            y_input.append(y_batch)
            x_batch = x_batcher.get_next_batch()
            y_batch = y_batcher.get_next_batch()
        x_input = np.vstack(x_input)
        y_input = np.vstack(y_input)
    else:
        x_input, y_input = convert_to_model_input(x, y, func, tokenizer)
    model.train(x_input, y_input, verbose=verbose, dim=dim)


def predict(x, y, model, tokenizer, dim=None):
    func = "get_individual_one_hot"
    if isinstance(model, NNFull):
        func = "get_one_hot"
    x, _ = convert_to_model_input(x, y, func, tokenizer)
    y_pred = model.predict(x, dim=dim)
    return y_pred


def report(y_pred: np.ndarray, y_true: np.ndarray, dim):
    """Generate plots showing the prediction accuracy on the given data."""
    # Calculate f1 score and plot
    f1 = []
    one_hot_y_true = np.array([get_individual_one_hot(ind_type) for ind_type in y_true])
    for i in range(len(y_pred[0])):
        idx = i
        if dim:
            idx = dim[i]
        one_axis_pred = y_pred[:, i]
        one_axis_true = one_hot_y_true[:, idx]
        temp = f1_score(one_axis_true, one_axis_pred, average="macro")
        f1.append(temp)
    plot_axis = ["I-E", "S-N", "F-T", "J-P"]
    if dim:
        temp = []
        for idx in dim:
            temp.append(plot_axis[idx])
        plot_axis = temp
    plt.plot(plot_axis, f1, "o")
    plt.xlabel("MBTI axis")
    plt.ylabel("Macro F1-score")
    plt.show()

    # Calculate prediction percentage of each individual class and in total, also plot.
    if dim:
        y_true = copy.copy(y_true)
        for j in range(len(y_true)):
            temp = ""
            for i in dim:
                temp += y_true[j][i]
            y_true[j] = temp
    y_pred = [get_type_from_individual_one_hot(encoding, dim) for encoding in y_pred]
    values, counts = np.unique(y_true, return_counts=True, axis=0)
    ind_val, ind_counts = np.unique([[list(word) for word in x] for x in y_true], return_counts=True)
    full_type = {}
    individual_dim = {}
    for i in range(len(y_pred)):
        pred_type = y_pred[i]
        true_type = y_true[i]
        if pred_type == true_type:
            if pred_type in full_type:
                full_type[pred_type] += 1
            else:
                full_type[pred_type] = 1
        for j in range(len(pred_type)):
            pred_axis = pred_type[j]
            true_axis = true_type[j]
            if pred_axis == true_axis:
                if pred_axis in individual_dim:
                    individual_dim[pred_axis] += 1
                else:
                    individual_dim[pred_axis] = 1
    x = []
    for i in range(len(values)):
        v = values[i]
        c = counts[i]
        if v in full_type:
            x.append(full_type[v] / c)
        else:
            x.append(0)
    plot_predictions(x, values, counts)
    x = []
    for i in range(len(ind_val)):
        v = ind_val[i]
        c = ind_counts[i]
        if v in individual_dim:
            x.append(individual_dim[v] / c)
        else:
            x.append(0)
    plot_predictions(x, ind_val, ind_counts, x_label="MBTI axis")
    return f1


def plot_predictions(x, val, count, x_label="MBTI type"):
    plt.subplot(2, 1, 1)
    plt.plot(val, x, "o")
    plt.ylim([0, plt.ylim()[1]])
    plt.xlabel(x_label)
    plt.ylabel("Correct prediction percentage")
    plt.subplot(2, 1, 2)
    plt.plot(val, count, "o")
    plt.ylim([0, plt.ylim()[1]])
    plt.xlabel(x_label)
    plt.ylabel("Number of examples")
    plt.tight_layout()
    plt.show()


def get_processed_data(
        size: int, preprocess: bool, folder="processed", two_gram: bool = False
):
    """Return a preprocessed data set, consisting of comments and types.
    :param size: If preprocess, determine the size of the dataset to process.
    :param preprocess: Determines whether to read from file or preprocess a new dataset.
    :param folder: Determine the folder within ./data to save to or read from.
    :param two_gram: Toggle two-gram tokens instead of one-gram."""
    print("Loading data...")
    if preprocess:
        types, comments = get_typed_comments(batch_size=int(size / 10), n=size)
        pre_process_batch(comments, int(len(comments) / 10), folder, two_gram=two_gram)
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


def tokenize_per_type():
    """Fit tokinizers on only comments from one specific type."""
    tokenizers = {}
    top_words = {}
    for mbti in MBTI_TYPES:
        print(f"{mbti}")
        idx = np.where(TYPES == mbti)
        if not len(idx[0]):
            continue
        comment_one_type = np.hstack(COMMENTS[idx])
        tokenizer = fit_tokenizer(data=comment_one_type)
        tokenizers[mbti] = tokenizer
        l = len(tokenizer.word_index)
        top_words[mbti] = [
            (
                tokenizer.index_word[i],
                round(tokenizer.word_counts[tokenizer.index_word[i]] / l, 2),
            )
            for i in range(1, 11)
        ]
    return tokenizers, top_words


def fit_tokenizer(data, num_words: int = 10000):
    """Generate a tokenizer from the given data.
    :param num_words: Only use top words when converting to matrix."""
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words)
    comment_batcher = BatchTracker(data_size=len(data), batch_size=int(len(data) / 10))
    batch = comment_batcher.get_next_batch()
    count = 0
    while len(batch):
        print(f"Tokenizer fitting batch on {count}")
        count += 1
        tokenizer.fit_on_texts(data[batch[0]:batch[1]])
        batch = comment_batcher.get_next_batch()
    return tokenizer


def plot_confusion_matrix(m, labels):
    cmd = ConfusionMatrixDisplay(m, display_labels=labels)
    cmd.plot(cmap="binary_r")
    plt.tight_layout()
    plt.show()


def similar_type(type_string: str, dim, tokenizer, model, comments, types):
    """ Train and report results when trained on only two specific types that differ in one axis. Also
     report results on unseen personality type in the differing axis. """
    cond = [type_string in x for x in TYPES]
    x_train, x_test, y_train, y_test = train_test_split(
        comments[cond], types[cond], test_size=0.25, random_state=1, stratify=None
    )
    train_model(
        x_train, y_train, tokenizer, model=model, verbose=True, dim=dim
    )

    # Show results on test set for trained personality types
    y_pred = predict(x_test, y_test, model, tokenizer, dim=dim)
    report(y_pred=y_pred, y_true=y_test, dim=dim)

    # Show results on unseen personality types
    y_pred = predict(COMMENTS[np.invert(cond)], TYPES[np.invert(cond)], model, tokenizer, dim=dim)
    report(y_pred=y_pred, y_true=TYPES[np.invert(cond)], dim=dim)


if __name__ == "__main__":
    START_TIME = time.time()
    COMMENTS, TYPES = get_processed_data(
        size=1000, preprocess=False, folder="test", two_gram=False
    )

    # TOKS, TOP_W = tokenize_per_type()

    TOKENIZER = fit_tokenizer(data=COMMENTS, num_words=10000)
    SAVE = False
    # FULL_MODEL = NNFull(save=save)

    INDIVIDUAL_MODELS = NNIndividual(save=SAVE, epoch=200)
    # INDIVIDUAL_MODELS = LGBM(save=save)

    similar_type("nfp", [0], TOKENIZER, INDIVIDUAL_MODELS, COMMENTS, TYPES)

    # X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    #     COMMENTS, TYPES, test_size=0.25, random_state=1, stratify=None
    # )
    # DIM = None
    # # train_model(x_train, y_train, TOKENIZER, model=FULL_MODEL)
    # train_model(
    #     X_TRAIN, Y_TRAIN, TOKENIZER, model=INDIVIDUAL_MODELS, verbose=True, dim=DIM
    # )
    # Y_PRED = predict(X_TEST, Y_TEST, INDIVIDUAL_MODELS, TOKENIZER, dim=DIM)
    # report(y_pred=Y_PRED, y_true=Y_TEST, dim=DIM)
    # # Y_PRED = predict(X_TRAIN, Y_TRAIN, INDIVIDUAL_MODELS, TOKENIZER, dim=DIM)
    # # Plot confusion matrix
    # # UNIQUE = np.unique(Y_TRAIN)
    # # STR_Y_PRED = [get_type_from_individual_one_hot(x) for x in Y_PRED]
    # # M = confusion_matrix(Y_TEST, STR_Y_PRED, labels=UNIQUE)
    # # plot_confusion_matrix(M, UNIQUE)
    print("--- %s seconds ---" % (time.time() - START_TIME))
