# engine.py
import torch
import torch.nn as nn


def train_one_epoch(data_loader, model, optimizer, device):
    # set the model to training mode
    model.train()

    # iterate through the batches
    for d in data_loader:
        # get the reviews and targets
        reviews = d["review"]
        targets = d["target"]
        # print(f"reviews shape: {reviews.size()}")
        # print(f"reviews: {reviews}")
        # move the data to cuda
        reviews = reviews.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        # print(f"targets: {targets.view(-1, 1)}")
        # clear the gradients
        optimizer.zero_grad()

        # make predictions
        predictions = model(reviews)
        # print(f"predictions: {predictions}")
        # calculate the loss
        loss = nn.BCEWithLogitsLoss()(predictions, targets.view(-1, 1))

        # backprop
        loss.backward()

        # optimizer step
        optimizer.step()


def evaluate(data_loader, model, device):

    # lists to store the predictions and targets
    final_predictions = []
    final_targets = []

    # set the model to evaluation mode
    model.eval()
    # disable gradients during inference
    with torch.no_grad():
        for d in data_loader:
            reviews = d["review"]
            targets = d["target"]

            # reviews, targets = (
            #     reviews.to(device, dtype=torch.long),
            #     targets.to(device, dtype=torch.float),
            # )

            reviews = reviews.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            # make predictions
            predictions = model(reviews)

            # move predictions and targets to list
            predictions = torch.sigmoid(predictions)
            predictions = predictions.cpu().numpy().tolist()
            targets = targets.cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)

    return final_predictions, final_targets
