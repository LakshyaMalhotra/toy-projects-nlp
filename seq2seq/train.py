import time
import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import *
from dataset import Fra2EngDataset
import model

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 10
CLIP = 1


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, data_loader, optimizer, criterion, clip, print_every=2500):
    epoch_loss = 0
    print_loss = 0
    model.train()

    for i, data in enumerate(data_loader):
        input_tensor = data["input"]
        target_tensor = data["output"]

        optimizer.zero_grad()

        output = model(input_tensor, target_tensor)
        # print(f"Output size: {output.size()}")
        # print(f"Target size: {target_tensor.size()}")

        output_dim = output.size(-1)
        output = output.view(-1, output_dim)
        target_tensor = target_tensor.squeeze()

        loss = criterion(output, target_tensor)
        # print(f"Loss: {loss}")

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        print_loss += loss.item()
        epoch_loss += loss.item()

        if (i + 1) % print_every == 0:
            print_avg_loss = print_loss / print_every
            print(f"Batch: {i + 1}, Loss: {print_avg_loss:.4f}")
            print_loss = 0

    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, criterion, print_every=100):
    epoch_loss = 0
    print_loss = 0

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input_tensor = data["input"]
            target_tensor = data["output"]

            output = model(input_tensor, target_tensor)
            # print(f"Output size: {output.size()}")
            # print(f"Target size: {target_tensor.size()}")

            output_dim = output.size(-1)
            output = output.view(-1, output_dim)
            target_tensor = target_tensor.squeeze()

            loss = criterion(output, target_tensor)
            # print(f"Loss: {loss}")

            epoch_loss += loss.item()
            print_loss += loss.item()

            if i % print_every == 0:
                print_avg_loss = print_loss / print_every
                print(f"Batch: {i + 1}, Loss: {print_avg_loss:.4f}")
                print_loss = 0
    return epoch_loss / len(data_loader)


if __name__ == "__main__":
    print("-" * 30)
    print("READING DATA AND PREPROCESSING")
    print("-" * 30)
    sentence_vectors = []

    # Get the input/output languages and sentence pairs
    input_lang, output_lang, pairs = prepare_data(
        lang1="eng", lang2="fra", reverse=True
    )

    # Create sentence vector pairs for input and output language pairs
    for pair in pairs:
        vectors = vector_from_pair(input_lang, output_lang, pair)
        sentence_vectors.append(vectors)

    # Instantiate train and test datasets
    dataset_train = Fra2EngDataset(sentence_vectors, device=DEVICE)
    dataset_test = Fra2EngDataset(sentence_vectors, device=DEVICE)

    # Split the dataset into train and test
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-460])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-460:])

    # Training and test dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False
    )

    # Define model specific parameters
    encoder_input_size = input_lang.n_words
    encoder_embed_dim = 100
    encoder_hidden_size = 128

    decoder_input_size = output_lang.n_words
    decoder_embed_dim = 100
    decoder_hidden_size = 128

    # Instantiate encoder and decoder
    encoder = model.EncoderRNN(
        input_size=encoder_input_size,
        embed_dim=encoder_embed_dim,
        hidden_size=encoder_hidden_size,
    )
    decoder = model.DecoderRNN(
        hidden_size=decoder_hidden_size,
        embed_dim=decoder_embed_dim,
        output_size=decoder_input_size,
    )

    # Define the model
    seq2seq_model = model.Seq2Seq(
        encoder, decoder, target_vocab_size=decoder_input_size, device=DEVICE
    ).to(DEVICE)
    print("-" * 13)
    print("MODEL SUMMARY")
    print("-" * 13)
    print(seq2seq_model)
    print(
        f"The model has {count_parameters(seq2seq_model)} trainable parameters."
    )
    print("-" * 23)
    print("TRAINING AND VALIDATION")
    print("-" * 23)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(seq2seq_model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):
        print(f"Epoch: {epoch+1}")

        start_time = time.time()
        print("-" * 8)
        print("TRAINING")
        print("-" * 8)
        train_loss = train(
            seq2seq_model, data_loader_train, optimizer, criterion, clip=CLIP
        )
        print("-" * 10)
        print("VALIDATING")
        print("-" * 10)
        valid_loss = evaluate(seq2seq_model, data_loader_test, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(seq2seq_model.state_dict(), "model_v1.pt")

        print(f"Time taken: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f},  Valid Loss: {valid_loss:.3f}")
