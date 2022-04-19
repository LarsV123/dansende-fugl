from models.model import Model
from sklearn.dummy import DummyClassifier
import numpy as np


class Baseline(Model):
    def __init__(self, save: bool, file_path: str = "./baseline", **kwargs):
        super(Baseline, self).__init__(save=save, file_path=file_path, **kwargs)

    def build_model(self):
        """Build and return a model meant for predicting one out of 16 personality types."""
        ie = DummyClassifier(strategy="most_frequent")
        sn = DummyClassifier(strategy="most_frequent")
        ft = DummyClassifier(strategy="most_frequent")
        jp = DummyClassifier(strategy="most_frequent")
        return [ie, sn, ft, jp]

    def train(self, x, y, verbose, dim=None):
        dimensions = ["I-E", "S-N", "F-T", "J-P"]
        model_idx = [i for i in range(len(self.model))]
        if dim:
            model_idx = dim
        for i in model_idx:
            print(f"\nTraining on dimension: {dimensions[i]}")
            model = self.model[i]
            labels = y[:, i]
            model.fit(x, labels, verbose)

    def predict(self, x, dim=None):
        model_idx = [i for i in range(len(self.model))]
        if dim:
            model_idx = dim
        full_type = []
        for i in model_idx:
            pred = self.model[i].predict(x)
            pred = np.array([[p] for p in pred])
            pred = np.where(pred <= 0.5, 0, 1)
            full_type.append(pred)
        return np.hstack(full_type)
