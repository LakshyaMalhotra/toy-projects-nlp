import re
import time
import string

import pandas as pd
from sklearn import decomposition
from sklearn import metrics
from sklearn import neural_network
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def preprocess_text(text):
    # replace all the punctuation with space
    clean_text = re.sub(rf"[{string.punctuation}]+", " ", text)
    clean_text = clean_text.split()
    clean_text = " ".join(clean_text)

    return clean_text


def mlp_lsa_model(fold, df, stop_words):
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
    svd = decomposition.TruncatedSVD(n_components=200)
    svd.fit(xtrain)
    xtrain_svd = svd.transform(xtrain)
    xvalid_svd = svd.transform(xvalid)

    ytrain = train_df.sentiment
    yvalid = valid_df.sentiment

    # instantiate neural network model
    mlp = neural_network.MLPClassifier(hidden_layer_sizes=(128, 64), verbose=True)

    # train the model on training reviews
    mlp.fit(X=xtrain_svd, y=ytrain)

    # make predictions on validation set
    preds = mlp.predict(xvalid_svd)
    preds_proba = mlp.predict_proba(xvalid_svd)[:, 1]

    # calculate accuracy and roc score
    accuracy = metrics.accuracy_score(yvalid, preds)
    roc = metrics.roc_auc_score(yvalid, preds_proba)

    valid_df["mlp_lsa_pred"] = preds_proba

    print(f"Accuracy: {accuracy:.4f},  ROC score: {roc:.4f}")
    print("")

    columns = ["id", "kfold", "sentiment", "mlp_lsa_pred"]

    return valid_df[columns]


if __name__ == "__main__":
    file_path = "../data/train_folds.csv"
    df = pd.read_csv(file_path)

    # preprocess the review column by removing the punctuations
    df["review"] = df["review"].apply(preprocess_text)

    stop_words = list(stopwords.words("english"))
    fold_outputs = []

    for f_ in range(5):
        start_time = time.time()
        print(f"Working on fold: {f_} ...")
        out_df = mlp_lsa_model(f_, df, stop_words)
        fold_outputs.append(out_df)

        print(f"Time elapsed: {(time.time() - start_time):.2f} seconds.")

    final_df = pd.concat(fold_outputs)
    final_df.to_csv("../outputs/mlp_lsa_model.csv", index=False)
    print(final_df.head())
    print(final_df.shape)
