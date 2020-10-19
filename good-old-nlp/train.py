# train.py
import io
import time
import torch
import h5py

import numpy as np
import pandas as pd

# using tensorflow for padding and other utils
import tensorflow as tf

from sklearn import metrics

import config
import dataset
import engine
import lstm


def load_vectors(fname):
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")

    # this following line is only needed for fasttext
    # n, d = map(int, fin.readline().split())

    # initialize a dict to store the embedding dictionary
    data = {}

    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = list(map(float, tokens[1:]))

    return data


def create_embedding_matrix(word_index, embedding_dict):
    # initialize the matrix with zeros
    embed_matrix = np.zeros((len(word_index) + 1, 300))

    # iterate over all the words
    for word, idx in word_index.items():
        # if word is found in pretrained embeddings, update the matrix
        # else the vector is of zeros
        if word in embedding_dict:
            embed_matrix[idx] = embedding_dict[word]

    return embed_matrix


def run(df, embed_matrix, fold):
    print(f"Fold: {fold}")
    print("=" * 7)
    # fetch training and validation dataframe
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)

    # using tf.keras for tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

    # convert training and validation data to sequences
    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xvalid = tokenizer.texts_to_sequences(valid_df.review.values)
    # print(f"xtrain shape: {np.array(xtrain).shape}")
    # print(f"xvalid shape: {np.array(xvalid).shape}")

    # zero pad the training and validation sequences to MAX_LEN
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(
        xtrain, maxlen=config.MAX_LEN
    )
    xvalid = tf.keras.preprocessing.sequence.pad_sequences(
        xvalid, maxlen=config.MAX_LEN
    )
    # print(xtrain)
    # print(f"xtrain shape: {np.array(xtrain).shape}")

    # get the training dataset
    train_dataset = dataset.IMDBDataset(
        reviews=xtrain, targets=train_df.sentiment.values
    )
    valid_dataset = dataset.IMDBDataset(
        reviews=xvalid, targets=valid_df.sentiment.values
    )

    # dataloaders for training and validation
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=2
    )

    # print("Loading embeddings ...")
    # embedding_dict = load_vectors("../data/glove.6B.300d.txt")
    # embed_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict)

    # use GPU if available
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # fetch the model
    model = lstm.LSTM(embedding_matrix=embed_matrix)

    # send model to the device
    model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )
    # Log results
    print(f"Training model on {device}...")
    # set best accuracy to zero
    best_accuracy = 0

    # early stopping counter to zero
    early_stopping_counter = 0
    early_stopping_patience = 7

    # train the model
    for epoch in range(config.EPOCHS):
        # start of the epoch
        start_time = time.time()

        # train one epoch
        engine.train_one_epoch(train_dataloader, model, optimizer, device)

        # validate
        predictions, targets = engine.evaluate(valid_dataloader, model, device)

        # applying the threshold to the predictions
        outputs = (np.array(predictions) >= 0.5).astype("int")

        # calculate accuracy
        accuracy = metrics.accuracy_score(targets, outputs)
        auc = metrics.roc_auc_score(targets, predictions)

        # step the scheduler
        scheduler.step(accuracy)

        # print some parameters
        print(f"Epoch: {epoch},  Accuracy: {accuracy:.5f},  ROC-AUC: {auc:.5f}")

        # simple early stopping
        if accuracy > best_accuracy:
            print(f"Accuracy improved: {best_accuracy:.6f} -> {accuracy:.6f}")
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_patience:
            print(
                f"Accuracy didn't improve for {early_stopping_patience} epochs, early stopping..."
            )
            break

        print(f"Time taken: {(time.time() - start_time):.2f} seconds.")


if __name__ == "__main__":
    # load data
    df = pd.read_csv("../data/train_folds.csv")
    # print(df.head())
    with h5py.File("../data/embedding_matrix_fasttext.h5", "r") as hf:
        embed_matrix = hf["dataset"][:]

    # train for all folds
    run(df, embed_matrix, fold=0)
    run(df, embed_matrix, fold=1)
    run(df, embed_matrix, fold=2)
    run(df, embed_matrix, fold=3)
    run(df, embed_matrix, fold=4)
