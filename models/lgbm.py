import lightgbm as lgb
import numpy as np
from models.model import Model


class LGBM(Model):
    def __init__(self, save: bool, file_path: str = "./LGBM_model", **kwargs):
        super(LGBM, self).__init__(save=save, file_path=file_path, **kwargs)

    def build_model(self):
        models = []
        for _ in range(4):
            model = lgb.LGBMClassifier(n_estimators=100, metric="binary_logloss", silent=True,
                                       n_jobs=4,
                                       learning_rate=0.1,
                                       colsample_bytree=0.7, max_depth=9, subsample=0.8)
            models.append(model)
        return models

    def train(self, x, y, verbose):
        dimensions = ["I-E", "S-N", "F-T", "J-P"]
        file_path = self.file_path + f"_{x.size}"
        for _ in range(len(self.model)):
            print(f"\nTraining on dimension: {dimensions[_]}")
            model = self.model[_]
            labels = y[:, _]
            self.model[_] = model.fit(x, labels)

    def predict(self, x):
        full_type = []
        for model in self.model:
            pred = model.predict(x)
            full_type.append(pred)
        temp = []
        for i in range(len(full_type[0])):
            temp.append([full_type[0][i], full_type[1][i], full_type[2][i], full_type[3][i]])
        return np.array(temp)
