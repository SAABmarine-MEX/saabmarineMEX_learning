import numpy as np
import pickle

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


class KNN:
    def __init__(self, k=5, weight=None):
        self.k = k
        self.weight = weight

    def fit(self, data_x, data_y, weight=None): # weight_tf = FunctionTransformer(lambda X: X * weight, validate=True)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=self.k, weights="uniform"))
        ])
        self.pipeline.fit(data_x, data_y)
        # scale, TODO: check if better scaling is needed


    def predict(self, data_x):
        return self.pipeline.predict(data_x)

    def save(self, file_path):
        if not file_path.endswith('.pkl'):
            file_path += '.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(self.pipeline, f)


    def tune_knn(self, X_train, y_train, X_val, y_val):
        best_rmse = np.inf
        best_cfg  = None

        for k in range(3,21):
            for p in (1,2):
                knn = Pipeline([
                    ("scaler", StandardScaler()),
                    ("knn",    KNeighborsRegressor(n_neighbors=k, p=p, weights="uniform"))
                ])
                knn.fit(X_train, y_train)
                yhat = knn.predict(X_val)
                rmse = np.sqrt(np.mean((yhat - y_val)**2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_cfg = (k,p)

        print(f"Best (k,p)={best_cfg} with RMSE={best_rmse:.4f}")

        # k_opt, p_opt = best_cfg
        # final_knn = Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("knn",    KNeighborsRegressor(
        #                 n_neighbors=k_opt,
        #                 p=p_opt,
        #                 weights="uniform"))
        #                     ])
        # final_knn.fit(X_train, y_train)
        # self.pipeline = final_knn

