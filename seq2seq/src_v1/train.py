import time
import math

import torch
import torch.nn as nn

import model
from utils import Preprocess
from engine import Engine

# count total trainable parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# calculate execution time
def epoch_time(start_time, end_time):
    elapsed_time = time.time() - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    N_EPOCHS = 10
    CLIP = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate preprocess class
    pp = Preprocess(verbose=True, batch_size=64)

    # create datasets and vocabularies
    pp.get_dataset()

    # get train and valid iterators
    train_iterator, valid_iterator = pp.batchify()

    # vocabulary sizes
    input_size = pp.input_size
    output_size = pp.target_size

    # input parameters to the model
    encoder_embed_dim = 500
    encoder_hidden_size = 1024

    decoder_embed_dim = 500
    decoder_hidden_size = 1024

    # define encoder and decoder
    encoder = model.Encoder(input_size, encoder_embed_dim, encoder_hidden_size)
    decoder = model.Decoder(output_size, decoder_embed_dim, decoder_hidden_size)

    # define the complete model
    net = model.Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    print(net)
    print(
        f"Total number of trainable parameters in the model: {count_params(net):,}"
    )

    # define the loss function and optimizer
    # we don't want to calculate loss on the padding index
    pad_idx = pp.target.vocab.stoi[pp.target.pad_token]
    criterion = nn.NLLLoss(ignore_index=pad_idx)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # instantiate engine class
    run = Engine(net, optimizer, criterion, DEVICE, CLIP)

    # best validation loss, used to save the best model
    best_valid_loss = float("inf")

    # iterate through all epochs
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        # training step
        train_loss = run.train(train_iterator)

        # validation step
        valid_loss = run.evaluate(valid_iterator)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(net.state_dict(), "lang_translation_1.pt")

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        print(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
        )
