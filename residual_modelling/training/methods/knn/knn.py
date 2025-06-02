import numpy as np
import pickle

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def train_knn(data_x, data_y, k, w):
    pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn",    KNeighborsRegressor(n_neighbors=k, weights="uniform"))
    ])
    pipeline.fit(data_x, data_y)
    # scale, TODO check if better scaling is needed

    return pipeline


def tune_knn(X_train, y_train, X_val, y_val, w):
    best_rmse = np.inf
    best_cfg  = None

    for k in range(3,11):
        for p in (1,2):
            knn = Pipeline([
                ("scaler", StandardScaler()),
                ("weight", w),
                ("knn",    KNeighborsRegressor(n_neighbors=k, p=p, weights="uniform"))
            ])
            knn.fit(X_train, y_train)
            yhat = knn.predict(X_val)
            rmse = np.sqrt(np.mean((yhat - y_val)**2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_cfg = (k,p)

    print(f"Best (k,p)={best_cfg} with RMSE={best_rmse:.4f}")

    k_opt, p_opt = best_cfg
    final_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsRegressor(
                    n_neighbors=k_opt,
                    p=p_opt,
                    weights="uniform"))
                        ])
    final_knn.fit(X_train, y_train)

    return final_knn


def main():
    k = 5  # Number of neighbors
    weight = np.array([1]*6 +[1]*6 + [1]*6, dtype=np.float32)
    weight_tf = FunctionTransformer(lambda X: X * weight, validate=True)

    print("Loading data...")
    data_dir = "../../data/data1/"
    data_file = "data.npz"
    data = np.load(data_dir + data_file)
    data_x = data["x"]
    data_y = data["y"]
    print("Data X shape:", data_x.shape)
    print("Data Y shape:", data_y.shape)



    print("Training KNN 3dof...")
    knn = train_knn(data_x, data_y, k, weight_tf)

    print("Saving knn 3dof")
    results_dir = "../../results/knn/"
    results_file = "knn_3dof.pkl"
    with open(results_dir + results_file, "wb") as f:
        pickle.dump(knn, f)



if __name__ == "__main__":
    main()
