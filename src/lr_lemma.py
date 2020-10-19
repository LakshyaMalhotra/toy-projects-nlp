import time
from functools import partial

import pandas as pd

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def lemmatize_text(lemma, text):
    return [lemma.lemmatize(w) for w in text.split()]


def lr_lemma_model(fold, df):
    # create train and valid df for given fold
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)

    # instantiate count vectorizer
    vectorizer = TfidfVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None,
        max_features=750,
        ngram_range=(1, 2),
    )

    # fit the count vectorizer on training reviews
    vectorizer.fit(train_df.review_lemmatized)

    # transform training and validation reviews
    xtrain = vectorizer.transform(train_df.review_lemmatized)
    xvalid = vectorizer.transform(valid_df.review_lemmatized)

    ytrain = train_df.sentiment
    yvalid = valid_df.sentiment

    # instantiate logistic regression model
    lr = linear_model.LogisticRegression(n_jobs=-1, max_iter=200)

    # train the model on training reviews
    lr.fit(X=xtrain, y=ytrain)

    # make predictions on validation set
    preds = lr.predict(xvalid)
    preds_proba = lr.predict_proba(xvalid)[:, 1]

    # calculate accuracy and roc score
    accuracy = metrics.accuracy_score(yvalid, preds)
    roc = metrics.roc_auc_score(yvalid, preds_proba)

    valid_df["lr_lemma_pred"] = preds_proba

    print(f"Accuracy: {accuracy:.4f},  ROC score: {roc:.4f}")
    print("")

    columns = ["id", "kfold", "sentiment", "lr_lemma_pred"]

    return valid_df[columns]


if __name__ == "__main__":
    file_path = "../data/train_folds.csv"

    # initialize the lemmatizer
    """
    Lemmatization keeps the meaning of the sentences intact.
    """
    lemma = WordNetLemmatizer()

    df = pd.read_csv(file_path)

    # lemmatize the review column
    df["review_lemmatized"] = df.review.apply(
        partial(lemmatize_text, lemma)
    ).apply(lambda x: " ".join(x))

    fold_outputs = []

    for f_ in range(5):
        start_time = time.time()
        print(f"Working on fold: {f_} ...")
        out_df = lr_lemma_model(f_, df)
        fold_outputs.append(out_df)

        print(f"Time elapsed: {(time.time() - start_time):.2f} seconds.")

    final_df = pd.concat(fold_outputs)
    final_df.to_csv("../outputs/lr_lemma_model.csv", index=False)
    print(final_df.head())
    print(final_df.shape)
