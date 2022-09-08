import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


if __name__ == "__main__":
    # load dataset
    emotions = load_dataset("emotion")

    print(emotions)
    train_ds = emotions["train"]
    print(train_ds.column_names)
    print(train_ds[15:17])

    # load model and its corresponding tokenizer
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # example text
    text = "Tokenizing text is a core task of NLP."

    # get the encoded text
    encoded_text = tokenizer(text)
    print(encoded_text)

    # get the tokens back from the encoded text
    tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
    print(tokens)

    # convert tokens to actual strings
    string = tokenizer.convert_tokens_to_string(tokens)
    print(string)

    # vocabulary size
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # maximum context size
    print(f"Max length of sequence for model: {tokenizer.model_max_length}")

    # fields that this particular model expects in the forward pass
    print(
        f"Input names expected by {model_ckpt}: {tokenizer.model_input_names}"
    )
