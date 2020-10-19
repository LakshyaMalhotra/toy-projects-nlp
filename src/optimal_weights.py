import glob
from functools import partial

import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.optimize import fmin


class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        X_coef = X * coef
        pred = np.sum(X_coef, axis=1)
        auc_score = metrics.roc_auc_score(y, pred)

        return -1 * auc_score

    def fit(self, X, y):
        partial_loss = partial(self._auc, X=X, y=y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        self.coef_ = fmin(partial_loss, init_coef, disp=True)

    def predict(self, X):
        X_coef = X * self.coef_
        preds = np.sum(X_coef, axis=1)
        return preds


def run_training(pred_df, fold):
    train_df = pred_df[pred_df["kfold"] != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df["kfold"] == fold].reset_index(drop=True)
    pred_cols = [
        "lr_cnt_pred",
        "lr_tfidf_pred",
        "lr_lemma_pred",
        "rf_lsa_pred",
        "mlp_lsa_pred",
    ]
    xtrain = train_df[pred_cols]
    xvalid = valid_df[pred_cols]

    opt = OptimizeAUC()
    opt.fit(xtrain, train_df.sentiment.values)
    preds = opt.predict(xvalid)
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"fold: {fold}, auc: {auc:.5f}")

    return opt.coef_


if __name__ == "__main__":
    files = glob.glob("../outputs/*.csv")
    df = None
    thresh = 0.5

    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on="id", how="left")

    # print(df.head())
    print(df.columns.tolist())
    print("")

    pred_cols = [
        "lr_cnt_pred",
        "lr_tfidf_pred",
        "lr_lemma_pred",
        "rf_lsa_pred",
        "mlp_lsa_pred",
    ]
    coeffs = []
    for j in range(5):
        coeffs.append(run_training(df, j))
    coeffs = np.array(coeffs)
    print(coeffs)
    best_coeffs = np.mean(coeffs, axis=0)
    print(best_coeffs)
    print(best_coeffs.sum())

    print("")
    print("Weighted average metrics:")
    avg_pred_probas = np.sum(df[pred_cols].values * best_coeffs, axis=1)
    avg_preds = (np.mean(df[pred_cols].values, axis=1) > thresh).astype("int")
    accuracy = metrics.accuracy_score(df.sentiment.values, avg_preds)
    auc = metrics.roc_auc_score(df.sentiment.values, avg_pred_probas)
    print(f"Accuracy: {accuracy:.5f}, ROC-AUC: {auc:.5f}")

    print("")
    print("Weighted rank average:")
    avg_pred_probas = np.sum(df[pred_cols].rank().values * best_coeffs, axis=1)
    avg_preds = (np.mean(df[pred_cols].values, axis=1) > thresh).astype("int")
    accuracy = metrics.accuracy_score(df.sentiment.values, avg_preds)
    auc = metrics.roc_auc_score(df.sentiment.values, avg_pred_probas)
    print(f"Accuracy: {accuracy:.5f}, ROC-AUC: {auc:.5f}")
