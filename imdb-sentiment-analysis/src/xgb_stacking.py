import glob
from functools import partial

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from scipy.optimize import fmin
import xgboost as xgb


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
    xtrain = train_df[pred_cols].values
    xvalid = valid_df[pred_cols].values

    clf = xgb.XGBClassifier()
    clf.fit(xtrain, train_df.sentiment.values)
    preds = clf.predict_proba(xvalid)[:, 1]
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"fold: {fold}, auc: {auc:.5f}")

    valid_df.loc[:, "xgb_pred"] = preds
    return valid_df


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
    dfs = []
    for j in range(5):
        dfs.append(run_training(df, j))

    final_valid_df = pd.concat(dfs)
    print(
        metrics.roc_auc_score(
            final_valid_df.sentiment.values, final_valid_df.xgb_pred.values
        )
    )