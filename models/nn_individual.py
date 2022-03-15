from keras import Sequential
from keras.layers import Dense, Dropout
from models.model import Model
from keras.metrics import AUC, Precision, Recall, TruePositives, TrueNegatives
import numpy as np
import matplotlib.pyplot as plt


class Individual:

    def __init__(self, layers: int, units: int, dropout: float, batch_size: int = 32, epochs: int = 20):
        model = Sequential()
        for i in range(layers):
            model.add(Dense(units, activation="relu", kernel_regularizer="L2"))
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["mean_squared_error", AUC(name="auc"), Precision(name="precision"), Recall(name="recall"),
                     TruePositives(name="true_positive"), TrueNegatives(name="true_negatives")],
        )
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, x, y, verbose, weights):
        return self.model.fit(
            x,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            verbose=verbose,
            class_weight=weights
        )

    def predict(self, y):
        return self.model.predict(y)


class NNIndividual(Model):
    def __init__(self, save: bool, file_path: str = "./individual_model", **kwargs):
        super(NNIndividual, self).__init__(save=save, file_path=file_path, **kwargs)

    def build_model(self):
        """Build and return four individual models, meant to predict one of the four axis in mbti each."""
        ie = Individual(layers=3, units=64, dropout=0.3, epochs=200)
        sn = Individual(layers=3, units=64, dropout=0.3, epochs=200)
        ft = Individual(layers=3, units=64, dropout=0.3, epochs=200)
        jp = Individual(layers=3, units=64, dropout=0.3, epochs=200)
        return [ie, sn, ft, jp]

    def train(self, x, y, verbose, dim=None):
        dimensions = ["I-E", "S-N", "F-T", "J-P"]
        file_path = self.file_path + f"_{x.size}"
        model_idx = [i for i in range(len(self.model))]
        if dim:
            model_idx = dim
        for i in model_idx:
            print(f"\nTraining on dimension: {dimensions[i]}")
            model = self.model[i]
            labels = y[:, i]
            val, count = np.unique(labels, return_counts=True)
            weights = {0: 1, 1: count[0] / count[1]}
            history = model.train(x, labels, verbose, weights)
            self.plot_metrics(history, dimensions[i])
            if self.save:
                model.model.save(file_path + f"_{dimensions[i]}")

    def predict(self, x, dim):
        model_idx = [i for i in range(len(self.model))]
        if dim:
            model_idx = dim
        full_type = []
        for i in model_idx:
            pred = self.model[i].predict(x)
            pred = np.where(pred <= 0.5, 0, 1)
            full_type.append(pred)
        return np.hstack(full_type)

    def plot_metrics(self, history, dim):
        metrics = ['loss', 'auc', "true_positive", "true_negatives"]
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.suptitle(f"{dim}")
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_' + metric],
                     color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.1, 1])
            else:
                plt.ylim([0, plt.ylim()[1]])
            plt.tight_layout()
            plt.legend()
        plt.show()
