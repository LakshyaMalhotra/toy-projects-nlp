# engine.py
import torch
import torch.nn as nn

from tqdm import tqdm


def loss_fn(outputs, targets):
    # return loss
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_one_epoch(data_loader, model, optimizer, scheduler, device):
    """
    Training function which trains for one epoch
    :param data_loader: pytorch dataloader object
    :param model: bert model
    :param optimizer: optimizer to use: adam, sgd, rmsprop etc
    :param scheduler: learning rate scheduler
    :param device: whether to use GPU or CPU
    """
    # put the model to train mode
    model.train()

    # iterate over all the batches
    for batch_idx, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        # extract targets, ids, mask and token_type_ids from the data loader
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets = d["targets"]

        # move all of these to device
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        # clear all the gradients in the optimzer
        optimizer.zero_grad()

        # get the model output
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        # calculate loss
        loss = loss_fn(outputs, targets)

        # backpropagate
        loss.backward()

        # if (batch_idx + 1) % accumulation_steps == 0:
        # step the optimizer
        optimizer.step()

        # step scheduler
        scheduler.step()


def evaluate(data_loader, model, device):
    """
    Perform inference on validation data
    :param data_loader: pytorch dataloader object
    :param model: bert model
    :param device: whether to use GPU or CPU
    """
    # put the model in evaluation mode
    model.eval()

    # empty lists to store targets and outputs
    final_targets = []
    final_outputs = []

    # don't backpropagate in evaluation
    with torch.no_grad():
        # iterate over all the batches
        for batch_idx, d in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            # extract targets, ids, mask and token_type_ids from the data loader
            ids = d["ids"]
            mask = d["mask"]
            token_type_ids = d["token_type_ids"]
            targets = d["targets"]

            # move all of these to device
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            # get the model output
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

            # add targets and predictions to the final lists
            targets = targets.cpu().numpy().tolist()
            final_targets.extend(targets)

            outputs = torch.sigmoid(outputs).cpu().numpy().tolist()
            final_outputs.extend(outputs)

    return final_outputs, final_targets