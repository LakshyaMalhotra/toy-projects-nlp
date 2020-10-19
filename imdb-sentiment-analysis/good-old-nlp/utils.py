# utils.py
"""
File to create the embedding matrix. It uses embedding dictionary either from 
fasttext or Glove. For the vocabulary, it uses the unique preprocessed words 
(no punctuations, lower-cased, etc.) as input. This embedding matrix is needed 
for the RNN since the embeddings weights are used from this matrix.
"""
import io
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf


def load_vectors(fname):
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")

    # this following line is only needed for fasttext
    n, d = map(int, fin.readline().split())

    # initialize a dict to store the embedding dictionary
    data = {}

    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = list(map(float, tokens[1:]))

    return data


def embedding_matrix(word_index, embedding_dict):
    # initialize the matrix with zeros
    embed_matrix = np.zeros((len(word_index) + 1, 300))

    # iterate over all the words
    for word, idx in word_index.items():
        # if word is found in pretrained embeddings, update the matrix
        # else the vector is of zeros
        if word in embedding_dict:
            embed_matrix[idx] = embedding_dict[word]

    return np.array(embed_matrix)


if __name__ == "__main__":
    file_name = "../data/crawl-300d-2M.vec"

    # read the sentiment dataframe
    print("Reading the CSV file...")
    df = pd.read_csv("../data/train_folds.csv")

    print("Tokenizing the dataframe...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())
    word_index = tokenizer.word_index

    # load embeddings into the memory
    print("Loading embeddings...")
    embedding_dict = load_vectors(file_name)

    # Getting the embedding matrix
    print("Creating embedding matrix...")
    embed_matrix = embedding_matrix(word_index, embedding_dict)

    # Writing the matrix to a file
    print("Writing the embedding matrix to a h5 file...")
    h5f = h5py.File("../data/embedding_matrix_fasttext.h5", "w")
    h5f.create_dataset("dataset", data=embed_matrix)
    h5f.close()
    print("done!")
