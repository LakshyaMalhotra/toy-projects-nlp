import random
import torch

import model
from utils import Preprocess


if __name__ == "__main__":
    DEVICE = torch.device("cpu")
    batch_size = 64

    # instantiate preprocess class
    pp = Preprocess(verbose=True, batch_size=batch_size)

    # create datasets and vocabularies
    pp.get_dataset()

    # get train and valid iterators
    _, valid_iterator = pp.batchify()

    # get the batch from valid iterator
    batch = next(iter(valid_iterator))

    # get random input and target tensor from the batch
    idx = random.choice(range(batch_size))
    print(idx)
    input_tensor = batch.src[:, idx]  # input_tensor shape: [seq_len]
    target_tensor = batch.trg[:, idx]  # target_tensor shape: [seq_len]

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
    net = model.Seq2Seq(encoder, decoder, DEVICE)

    # load the saved model
    net.load_state_dict(
        torch.load("lang_translation_1.pt", map_location=DEVICE)
    )

    # put the model to evaluation mode
    net.eval()

    # reshape the input and target to feed them in the model
    inp = input_tensor.view(-1, 1)  # input_tensor shape: [seq_len, 1]
    targ = target_tensor.view(-1, 1)  # target_tensor shape: [seq_len, 1]

    # perform inference
    with torch.no_grad():
        output = net(inp, targ)

    # create input, target and output arrays from the tensors
    input_arr = input_tensor.cpu().numpy().tolist()[::-1]
    input_arr = [val for val in input_arr if val not in range(4)]
    target_arr = target_tensor.cpu().numpy().tolist()
    target_arr = [val for val in target_arr if val not in range(4)]
    output = output.squeeze()
    output_arr = output.argmax(1).cpu().numpy().tolist()
    output_arr = [val for val in output_arr if val not in range(4)]

    # create sentences from indexes
    input_sent = " ".join([pp.source.vocab.itos[val] for val in input_arr])
    target_sent = " ".join([pp.target.vocab.itos[val] for val in target_arr])
    output_sent = " ".join([pp.target.vocab.itos[val] for val in output_arr])

    # print results
    print(f"Input sentence:  {input_sent}")
    print(f"Target sentence: {target_sent}")
    print(f"Model output:    {output_sent}")
