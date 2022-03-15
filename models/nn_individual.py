from keras import Sequential
from keras.layers import Dense, Dropout
from models.model import Model


class NNIndividual(Model):
    def __init__(self, save: bool, file_path: str = "./individual_model", **kwargs):
        super(NNIndividual, self).__init__(save=save, file_path=file_path, **kwargs)

    def build_model(self):
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
                metrics=["mean_squared_error", "accuracy"],
            )
            models.append(model)
        return models

    def train(self, x, y, verbose):
        dimensions = ["I-E", "S-N", "F-T", "J-P"]
        file_path = self.file_path + f"_{x.size}"
        for _ in range(len(self.models)):
            print(f"\nTraining on dimension: {dimensions[_]}")
            model = self.models[_]
            model.fit(
                x,
                y[:, _],
                batch_size=self.batch_size,
                epochs=self.epoch,
                validation_split=0.2,
                verbose=verbose,
            )
            if self.save:
                model.save(file_path + f"_{dimensions[_]}")
