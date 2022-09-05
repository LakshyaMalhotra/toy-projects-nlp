import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import PreProcess
from dataset import NameDataset
from model import RNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    model,
    data_loader,
    criterion,
    optimizer,
    device=device,
    print_every=5000,
    plot_every=500,
):
    # setting the model to train mode
    model.train()

    epoch_loss = 0
    plot_loss = 0
    print_loss = 0

    losses = []

    # iterating through the train_dataloader
    for it, data in enumerate(data_loader):
        input_tensor = data[0]
        target_tensor = data[1]

        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        optimizer.zero_grad()

        out = model(input_tensor)

        loss = criterion(out, target_tensor.squeeze(0))

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        print_loss += loss.item()
        plot_loss += loss.item()

        if (it + 1) % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(f"Train batch iter: {it +1}, Loss: {print_loss_avg:.4f}")
            print_loss = 0

        if (it + 1) % plot_every == 0:
            plot_loss_avg = plot_loss / plot_every
            losses.append(plot_loss_avg)
            plot_loss = 0

    return epoch_loss / len(data_loader), losses


def evaluate(
    model, data_loader, criterion, device=device, print_every=200,
):
    # setting the model to train mode
    model.eval()

    epoch_loss = 0
    print_loss = 0
    with torch.no_grad():
        # iterating through the valid_dataloader
        for it, data in enumerate(data_loader):
            input_tensor = data[0]
            target_tensor = data[1]

            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            out = model(input_tensor)

            loss = criterion(out, target_tensor.squeeze(0))

            epoch_loss += loss.item()
            print_loss += loss.item()

            if (it + 1) % print_every == 0:
                print_loss_avg = print_loss / print_every
                print(f"Valid batch iter: {it +1}, Loss: {print_loss_avg:.4f}")
                # print(input_tensor, target_tensor)
                input_name, input_lang, pred_lang = show_predictions(
                    input_tensor, target_tensor, out
                )
                print(
                    f"Input name: {input_name}, input language: {input_lang}, predicted language: {pred_lang}"
                )
                print_loss = 0

    return epoch_loss / len(data_loader)


def show_predictions(input_tensor, target_tensor, prediction):
    input_array = input_tensor.squeeze().numpy().tolist()
    target = target_tensor.squeeze().numpy().tolist()
    input_name = "".join([idx2char[idx] for idx in input_array])
    target_name = all_categories[target]
    pred_idx = prediction.argmax(1).numpy().tolist()[0]
    pred_lang = all_categories[pred_idx]

    return input_name, target_name, pred_lang


if __name__ == "__main__":
    num_epochs = 10
    # define the path to input files
    path = "names/*.txt"

    # instantiate the preprocess class
    preprocess = PreProcess()

    # read the files and preprocess
    preprocess.read_and_preprocess(path=path)

    # get all categories and names
    all_categories = preprocess.all_categories
    all_names = preprocess.all_names
    all_letters = preprocess.all_letters
    char2idx = preprocess.char2idx
    idx2char = preprocess.idx2char
    n_letters = preprocess.n_letters
    n_categories = len(all_categories)

    print(all_letters)
    print(n_letters)
    print(char2idx)

    # create pytorch datasets for training and validation
    dataset_train = NameDataset(
        all_categories=all_categories, names=all_names, char2idx=char2idx
    )
    dataset_valid = NameDataset(
        all_categories=all_categories, names=all_names, char2idx=char2idx
    )

    # split the dataset to train and valid
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-500])
    dataset_valid = torch.utils.data.Subset(dataset_valid, indices[-500:])

    # define train and valid dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False
    )

    # data = dataset_train.__getitem__(56)
    # print(data)
    # data = next(iter(data_loader_train))
    # print(data)

    # define the model parameters
    input_size = n_letters
    hidden_size = 128
    output_size = n_categories
    embed_dim = 100

    # instantiate the model
    model = RNN(
        input_size=input_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        output_size=output_size,
    )
    model = model.to(device)
    print(model)
    print(
        f"There are {count_params(model):,} trainable parameters in the model."
    )
    # define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # store the losses to plot
    plot_losses = []
    best_valid_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_epoch_loss, losses = train(
            model, data_loader_train, criterion, optimizer, device=device
        )
        print(f"Epoch: {epoch}, Overall train loss: {train_epoch_loss:.4f}")
        print("-" * 60)
        valid_epoch_loss = evaluate(model, data_loader_valid, criterion)
        if valid_epoch_loss < best_valid_loss:
            print("Validation loss improved, saving model...")
            torch.save(model.state_dict(), "models/text-classification_v1.pt")
            best_valid_loss = valid_epoch_loss

        print(f"Epoch: {epoch}, Overall valid loss: {valid_epoch_loss:.4f}")
        print("-" * 60)
        plot_losses.extend(losses)

    # plot results
    fig = plt.figure(figsize=(8, 6))
    plt.plot(plot_losses)
    plt.show()
