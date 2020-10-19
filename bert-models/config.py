# config.py
import transformers

# maximum number of tokens in a sentence
MAX_LEN = 512

# using smaller batch sizes because model is huge
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2

# train for 10 epochs
EPOCHS = 10

# path to BERT model files
BERT_PATH = "../data/bert-base-uncased"

# path for saving the model
MODEL_PATH = "../models/bert_model.bin"

# training file
TRAINING_FILE = "../data/train_folds.csv"

# define the tokenizer
# use tokenizer and model from huggingface's transformers
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    f"{BERT_PATH}/vocab.txt", do_lower_case=True
)
