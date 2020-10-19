# train.py
import time
import config
import dataset
import engine
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run(df, fold):
    """
    Function to train the model
    """
    print(f"Fold: {fold}")
    print("=" * 7)
    # fetch training and validation dataframe
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)

    # get the training dataset
    train_dataset = dataset.BERTDataset(
        reviews=train_df.review.values, targets=train_df.sentiment.values
    )
    valid_dataset = dataset.BERTDataset(
        reviews=valid_df.review.values, targets=valid_df.sentiment.values
    )

    # dataloaders for training and validation
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=2
    )

    # use GPU if available
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # fetch the model
    model = BERTBaseUncased()

    # send model to the device
    model.to(device)

    # create parameters we want to optimize; we generally don't use any weight
    # decay for bias and weight layers
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # calculate the number of training steps
    num_train_steps = int(
        len(train_df) / config.TRAIN_BATCH_SIZE * config.EPOCHS
    )

    # optimizer; provided by transformers
    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    # learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # Log results
    print(f"Training model on {device}...")
    # set best accuracy to zero
    best_accuracy = 0

    # early stopping counter to zero
    early_stopping_counter = 0
    early_stopping_patience = 8

    # train the model
    for epoch in range(config.EPOCHS):
        # start of the epoch
        start_time = time.time()

        # train one epoch
        engine.train_one_epoch(
            train_dataloader, model, optimizer, scheduler, device
        )

        # validate
        predictions, targets = engine.evaluate(valid_dataloader, model, device)

        # applying the threshold to the predictions
        outputs = (np.array(predictions) >= 0.5).astype("int")

        # calculate accuracy
        accuracy = metrics.accuracy_score(targets, outputs)
        auc = metrics.roc_auc_score(targets, predictions)

        # print some parameters
        print(f"Epoch: {epoch},  Accuracy: {accuracy:.5f},  ROC-AUC: {auc:.5f}")

        # simple early stopping
        if accuracy > best_accuracy:
            print(f"Accuracy improved: {best_accuracy:.6f} -> {accuracy:.6f}")
            torch.save(model.state_dict(), config.MODEL_PATH)
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

    # train for all folds
    run(df, fold=0)
    # run(df, fold=1)
    # run(df, fold=2)
    # run(df, fold=3)
    # run(df, fold=4)
