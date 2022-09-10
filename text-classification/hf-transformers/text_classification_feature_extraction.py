import torch
import numpy as np
import datasets as hfd
from transformers import AutoModel, AutoTokenizer


def extract_hidden_states(batch):
    inputs = {
        k: v.to(device)
        for k, v in batch.items()
        if k in tokenizer.model_input_names
    }

    # extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    # return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


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
    model_ckpt = "distilbert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_ckpt).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    text = "this is a test"
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids.squeeze(0))
    print(tokens)
    print(f"Input tensor shape: {inputs['input_ids'].size()}")

    # add inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(inputs)

    # load dataset
    emotions = hfd.load_dataset("emotion")
    emotions_encoded = tokenize_dataset(emotions)
    emotions_encoded.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )
    emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

    print(emotions_hidden["train"].column_names)

    # create feature matrix
    X_train = np.array(emotions_hidden["train"]["hidden_state"])
    X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
    y_train = np.array(emotions_hidden["train"]["label"])
    y_valid = np.array(emotions_hidden["validation"]["label"])

    print(X_train.shape, X_valid.shape)
