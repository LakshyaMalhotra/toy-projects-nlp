import io
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def load_vectors(fname):
    fin = io.open(
        fname, mode="r", encoding="utf-8", newline="\n", errors="ignore"
    )
    # n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = list(map(float, tokens[1:]))

    return data


def sentence_to_vec(sent, embedding_dict, stop_words, tokenizer):
    # convert the sentence to string and lowercase it
    words = str(sent).lower()

    # tokenize the sentence
    words = tokenizer(words)

    # remove the stopwords
    words = [w for w in words if w not in stop_words]

    # keep only alpha-numeric tokens
    words = [w for w in words if w.isalpha()]

    # empty list to store word_embeddings
    word_embeddings = []
    for w in words:
        # for every word, fetch the embedding vector and
        # append it in the word embedding list
        if w in embedding_dict:
            word_embeddings.append(embedding_dict[w])

    # if we don't have any words, return vector of zeros
    if len(word_embeddings) == 0:
        return np.zeros(300)

    # convert the embeddings to array
    word_embeddings = np.array(word_embeddings)

    # calculate the sum over the rows
    sent_vec = np.sum(word_embeddings, axis=0)

    # return the normalized vector
    return sent_vec / np.linalg.norm(sent_vec)


if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("../data/train_folds.csv")
    print(df.head(2))

    # load embeddings into the memory
    print("Loading embeddings...")
    embeddings = load_vectors("../data/glove.6B.300d.txt")

    # create sentence embeddings
    print("Creating sentence vectors...")
    vectors = []
    for review in df.review.values:
        sent_vec = sentence_to_vec(
            review, embeddings, stop_words=[], tokenizer=word_tokenize
        )
        vectors.append(sent_vec)

    vectors = np.array(vectors)

    # fetch lables
    y = df.sentiment.values

    # initiate the stratified kfold
    skf = model_selection.StratifiedKFold(n_splits=5)

    # iterate through the folds
    for f_, (t_, v_) in enumerate(skf.split(X=vectors, y=y)):
        print(f"Training fold: {f_}")
        xtrain = vectors[t_, :]
        ytrain = y[t_]

        xvalid = vectors[v_, :]
        yvalid = y[v_]

        # initialize logistic regression model
        model = linear_model.LogisticRegression()

        # fit the model
        model.fit(xtrain, ytrain)

        # make predictions on the validation data
        preds = model.predict(xvalid)
        pred_proba = model.predict_proba(xvalid)[:, 1]

        # calculate accuracy and ROC score
        accuracy = metrics.accuracy_score(yvalid, preds)
        auc = metrics.roc_auc_score(yvalid, pred_proba)

        print(f"Accuracy: {accuracy:.5f}, ROC-AUC: {auc:.5f}")
        print("")
