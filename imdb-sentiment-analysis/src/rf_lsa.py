import re
import time
import string

import pandas as pd
from sklearn import decomposition
from sklearn import metrics
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def preprocess_text(text):
    # replace all the punctuation with space
    clean_text = re.sub(rf"[{string.punctuation}]+", " ", text)
    clean_text = clean_text.split()
    clean_text = " ".join(clean_text)

    return clean_text


def rf_lsa_model(fold, df, stop_words):
    # create train and valid df for given fold
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)

    # instantiate count vectorizer
    vectorizer = TfidfVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None,
        max_features=5000,
        stop_words="english",
    )

    # fit the count vectorizer on training reviews
    vectorizer.fit(train_df.review.values)

    # transform training and validation reviews
    xtrain = vectorizer.transform(train_df.review.values)
    xvalid = vectorizer.transform(valid_df.review.values)

    # Use SVD to reduce the dimensionality of the sparse matrix
    svd = decomposition.TruncatedSVD(n_components=150)
    svd.fit(xtrain)
    xtrain_svd = svd.transform(xtrain)
    xvalid_svd = svd.transform(xvalid)

    ytrain = train_df.sentiment
    yvalid = valid_df.sentiment

    print(
        f"xtrain and ytrain shape after svd: {xtrain_svd.shape}, {ytrain.shape}"
    )
    print(
        f"xvalid and yvalid shape after svd: {xvalid_svd.shape}, {yvalid.shape}"
    )
    # instantiate random forest model
    rf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)

    # train the model on training reviews
    rf.fit(X=xtrain_svd, y=ytrain)

    # make predictions on validation set
    preds = rf.predict(xvalid_svd)
    preds_proba = rf.predict_proba(xvalid_svd)[:, 1]

    # calculate accuracy and roc score
    accuracy = metrics.accuracy_score(yvalid, preds)
    roc = metrics.roc_auc_score(yvalid, preds_proba)

    valid_df["rf_lsa_pred"] = preds_proba

    print(f"Accuracy: {accuracy:.4f},  ROC score: {roc:.4f}")
    print("")

    columns = ["id", "kfold", "sentiment", "rf_lsa_pred"]

    return valid_df[columns]


if __name__ == "__main__":
    file_path = "../data/train_folds.csv"
    df = pd.read_csv(file_path)

    # preprocess the review column by removing the punctuations
    df["review"] = df["review"].apply(preprocess_text)

    stop_words = list(stopwords.words("english"))

    text = "I am tired!/:( I like fruits... &and, umm..@@milk?"
    clean_text = preprocess_text(text)
    print(text + "\n" + clean_text)

    fold_outputs = []

    for f_ in range(5):
        start_time = time.time()
        print(f"Working on fold: {f_} ...")
        out_df = rf_lsa_model(f_, df, stop_words)
        fold_outputs.append(out_df)

        print(f"Time elapsed: {(time.time() - start_time):.2f} seconds.")

    final_df = pd.concat(fold_outputs)
    final_df.to_csv("../outputs/rf_lsa_model.csv", index=False)
    print(final_df.head())
    print(final_df.shape)
