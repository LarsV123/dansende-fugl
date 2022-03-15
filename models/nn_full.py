from keras.layers import Dense, Dropout
from keras.models import Sequential
from models.model import Model


class NNFull(Model):
    def __init__(self, save: bool, file_path: str = "./full_model", **kwargs):
        super(NNFull, self).__init__(save=save, file_path=file_path, **kwargs)

    def build_model(self):
        """Build and return a model meant for predicting one out of 16 personality types."""
        model = Sequential()
        for i in range(4):
            model.add(Dense(64, activation="relu", kernel_regularizer="L2"))
            model.add(Dropout(0.2))
        model.add(Dense(16, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["categorical_accuracy", "mean_squared_error"],
        )
        return [model]

    def train(self, x, y, verbose):
        file_path = self.file_path + f"_{x.size}"
        for model in self.models:
            model.fit(
                x,
                y,
                batch_size=self.batch_size,
                epochs=self.epoch,
                validation_split=0.2,
                verbose=verbose,
            )
            if self.save:
                model.save(file_path)
