import glob
import pandas as pd
import numpy as np
from sklearn import metrics

if __name__ == "__main__":
    files = glob.glob("../outputs/*.csv")
    df = None
    thresh = 0.50

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

    for col in pred_cols:
        preds = (df[col] > thresh).astype("int").values
        accuracy = metrics.accuracy_score(df.sentiment.values, preds)
        auc = metrics.roc_auc_score(df.sentiment.values, df[col].values)
        print(f"pred col: {col},  Accuracy: {accuracy},  ROC-AUC: {auc:.5f}")

    print("")
    print("Average metrics:")
    avg_pred_probas = np.mean(df[pred_cols].values, axis=1)
    avg_preds = (avg_pred_probas > thresh).astype("int")
    accuracy = metrics.accuracy_score(df.sentiment.values, avg_preds)
    auc = metrics.roc_auc_score(df.sentiment.values, avg_pred_probas)
    print(f"Accuracy: {accuracy:.5f}, ROC-AUC: {auc:.5f}")

    print("")
    print("Weighted average metrics:")
    avg_pred_probas = np.average(
        df[pred_cols].values, weights=[2, 3, 1.5, 1, 1.5], axis=1
    )
    avg_preds = (avg_pred_probas > thresh).astype("int")
    accuracy = metrics.accuracy_score(df.sentiment.values, avg_preds)
    auc = metrics.roc_auc_score(df.sentiment.values, avg_pred_probas)
    print(f"Accuracy: {accuracy:.5f}, ROC-AUC: {auc:.5f}")

    print("")
    print("Rank average:")
    avg_pred_probas = np.mean(df[pred_cols].rank().values, axis=1)
    avg_preds = (np.mean(df[pred_cols].values, axis=1) > thresh).astype("int")
    accuracy = metrics.accuracy_score(df.sentiment.values, avg_preds)
    auc = metrics.roc_auc_score(df.sentiment.values, avg_pred_probas)
    print(f"Accuracy: {accuracy:.5f}, ROC-AUC: {auc:.5f}")

    print("")
    print("Weighted rank average:")
    avg_pred_probas = np.average(
        df[pred_cols].rank().values, weights=[2, 1.5, 1.5, 1, 3], axis=1
    )
    avg_preds = (np.mean(df[pred_cols].values, axis=1) > thresh).astype("int")
    accuracy = metrics.accuracy_score(df.sentiment.values, avg_preds)
    auc = metrics.roc_auc_score(df.sentiment.values, avg_pred_probas)
    print(f"Accuracy: {accuracy:.5f}, ROC-AUC: {auc:.5f}")
