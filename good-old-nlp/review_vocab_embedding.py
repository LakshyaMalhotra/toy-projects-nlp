import io
import numpy as np
import pandas as pd

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


def sentence_to_vec(sent, embedding_dict, tokenizer, stop_words=None):
    # convert the sentence to string and lowercase it
    words = str(sent).lower()

    # tokenize the sentence
    words = tokenizer(words)

    # remove the stopwords
    if stop_words is not None:
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
    embeddings = load_vectors("../data/glove.42B.300d.txt")

    # create sentence embeddings
    print("Creating sentence vectors...")
    vectors = []
    for review in df.review.values:
        sent_vec = sentence_to_vec(review, embeddings, tokenizer=word_tokenize)
        vectors.append(sent_vec)

    vectors = np.array(vectors)

    # fetch labels
    ids = df.id.values
    y = df.sentiment.values
    fold = df.kfold.values
    column_values = list(zip(ids, vectors, y, fold))
    sentence_vec_df = pd.DataFrame(
        column_values,
        columns=["id", "sent_vec", "sentiment", "kfold"],
    )
    sentence_vec_df.to_csv(
        "../data/sentence_embeddings_glove42B_300d.csv", index=False
    )
