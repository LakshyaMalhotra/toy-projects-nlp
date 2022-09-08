import datasets as hfd
from transformers import AutoTokenizer


def subword_tokenization_demo(tokenizer: AutoTokenizer):
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


def tokenize_dataset(data: hfd.DatasetDict):
    """`batched=True` will encode the data in batches. `batch_size=None` means
    that the tokenization will be applied to the whole dataset as a
    single batch. This ensures the `input_ids` and `attention_mask` have the
    same shape globally.
    """
    data_encoded = data.map(tokenize, batched=True, batch_size=None)
    return data_encoded


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


if __name__ == "__main__":
    # load dataset
    emotions = hfd.load_dataset("emotion")

    print(emotions)
    train_ds = emotions["train"]
    print(train_ds.column_names)
    print(train_ds[15:17])

    # load model and its corresponding tokenizer
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # subword tokenization
    subword_tokenization_demo(tokenizer)

    # tokenize demo for a small batch from the dataset
    batch_tokens = tokenize(train_ds[:2])
    print(batch_tokens)

    # tokenize the whole dataset
    emotions_encoded = tokenize_dataset(emotions)
    print(emotions_encoded["train"].column_names)
