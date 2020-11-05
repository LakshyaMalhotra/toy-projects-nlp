import time
import math

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from dataset import Fra2EngDataset
import model

DEVICE = torch.device("cpu")
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

    # Sample random sentence pair
    idx = np.random.choice(len(sentence_vectors))
    input_vector = sentence_vectors[idx][0]
    target_vector = sentence_vectors[idx][1]

    # Converting the vectors to torch tensors
    input_tensor = torch.LongTensor(input_vector)
    target_tensor = torch.LongTensor(target_vector)

    input_tensor = input_tensor.unsqueeze(0)
    target_tensor = target_tensor.unsqueeze(0)

    print(input_tensor)
    print(target_tensor)

    # Define model specific parameters
    encoder_input_size = input_lang.n_words
    encoder_embed_dim = 300
    encoder_hidden_size = 512

    decoder_input_size = output_lang.n_words
    decoder_embed_dim = 300
    decoder_hidden_size = 512

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
        encoder, decoder, target_vocab_size=decoder_input_size
    )

    # Load the model checkpoint
    seq2seq_model.load_state_dict(
        torch.load("models/model_v2.pt", map_location=DEVICE)
    )

    # Get predictions
    seq2seq_model.eval()

    with torch.no_grad():
        output = seq2seq_model(input_tensor, target_tensor)
        output_dim = output.size(-1)
        output = output.view(-1, output_dim)
        _, top1 = output.topk(1, dim=1)
        top1 = top1.squeeze()

    print(output.size())
    print(top1)

    print(
        f"Input sentence: {' '.join([input_lang.idx2word[idx] for idx in input_vector])}"
    )
    print(
        f"Target sentence: {' '.join([output_lang.idx2word[idx] for idx in target_vector])}"
    )
    print(
        f"Model output: {' '.join([output_lang.idx2word[idx] for idx in top1.tolist()])}"
    )
