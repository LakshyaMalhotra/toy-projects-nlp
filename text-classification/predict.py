import random
import torch


from utils import PreProcess
from dataset import NameDataset
from model import RNN

device = torch.device("cpu")

# show predictions from inferences
def show_predictions(input_tensor, target_tensor, prediction):
    input_array = input_tensor.squeeze().numpy().tolist()
    target = target_tensor.squeeze().numpy().tolist()
    input_name = "".join([idx2char[idx] for idx in input_array])
    target_name = all_categories[target]
    pred_idx = prediction.argmax(1).numpy().tolist()[0]
    pred_lang = all_categories[pred_idx]

    return input_name, target_name, pred_lang


# make inferences
def predict(model, dataset, random_choice=42):
    model.eval()
    input_tensor, target_tensor = dataset.__getitem__(random_choice)
    input_tensor = input_tensor.view(1, -1)
    target_tensor = target_tensor.view(1, -1)

    with torch.no_grad():
        pred = model(input_tensor)

    input_name, target_name, pred_lang = show_predictions(
        input_tensor, target_tensor, pred
    )

    return input_name, target_name, pred_lang


if __name__ == "__main__":
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

    model.load_state_dict(
        torch.load("models/text-classification_v1.pt", map_location="cpu")
    )

    dataset = NameDataset(
        all_categories=all_categories, names=all_names, char2idx=char2idx
    )

    choice = random.choice(range(len(dataset)))
    input_name, input_lang, pred_lang = predict(model, dataset, choice)
    print(
        f"Input name: {input_name}, input language: {input_lang}, predicted language: {pred_lang}"
    )

