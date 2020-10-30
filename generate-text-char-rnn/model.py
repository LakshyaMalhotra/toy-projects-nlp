# Library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

# RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=False,
            num_layers=n_layers,
            bidirectional=False,
        )
        self.drop = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)

        out = torch.cat((avg_pool, max_pool), 1)
        out = self.drop(out)
        out = F.relu(self.linear1(out))
        out = self.drop(out)
        out = self.linear2(out)

        return out, hidden

    def init_hidden(self, device, batch_size=1):
        hidden = torch.zeros(1, batch_size, self.hidden_size)

        return hidden


def train(
    model,
    cat_tensor,
    input_line_tensor,
    target_line_tensor,
    criterion,
    optimizer,
    device,
):
    model.train()
    target_line_tensor.unsqueeze_(-1)
    hidden = model.init_hidden(device, batch_size=1)

    optimizer.zero_grad()

    total_loss = 0

    ## The heart of the network, the place where I most struggled with
    # The network expects input to be of form: (batch_size, sequence length, input features)
    # Since we are passing one hot encoded letter at a time so in our case the
    # input would be (1, 1, n_letters). Also, while computing loss, the target
    # should be of shape [..., 1]
    ## Sending the input and output to GPU
    input_line_tensor = input_line_tensor.to(device)
    target_line_tensor = target_line_tensor.to(device)

    # Iterating through the word
    for i in range(input_line_tensor.size(0)):
        # in_tensor = input_line_tensor[:, i, :].unsqueeze(0)
        in_tensor = input_line_tensor[i].unsqueeze(0)
        output, hidden = model(in_tensor, hidden)
        loss = criterion(output, target_line_tensor[i])
        total_loss += loss

    # back prop
    total_loss.backward()

    # optimizer update
    optimizer.step()

    return output, total_loss / input_line_tensor.size(0)


if __name__ == "__main__":
    # read all the files
    files = find_files("data/names/*.txt")
    print(files)

    # vocabulary
    all_letters = string.ascii_letters + ".,;'"
    print(f"Letters: {all_letters}")

    # total number of letters
    n_letters = len(all_letters) + 1  # for EOS
    print(f"Total number of letters: {n_letters}")

    # check if the word normalization is working properly
    print(unicode_to_ascii("Schrödinger"))

    # Build a category line dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    # print(read_lines("data/names/German.txt"))

    letters_list = []
    for file in files:
        category = os.path.splitext(os.path.basename(file))[0]
        all_categories.append(category)
        lines = read_lines(file)
        category_lines[category] = lines
        for name in lines:
            for c in name:
                letters_list.append(c)
    n_categories = len(all_categories)
    print(f"Total letters: {len(letters_list)}")
    print(f"Number of unique letters: {len(set(letters_list))}")
    print(f"Categories: {all_categories}")

    print("Generating a random training example...")
    (
        cat,
        line,
        cat_tensor,
        in_line_tensor,
        target_line_tensor,
    ) = random_training_example()

    print(f"Category: {cat}")
    print(f"Word: {line}")
    print(f"Category tensor: {cat_tensor}")
    print(f"Target line tensor: {target_line_tensor}")
    print(f"Input line tensor shape: {in_line_tensor.shape}")
    # for i in range(in_line_tensor.shape[1]):
    #     print(in_line_tensor[:, i, :].unsqueeze(0).shape)
    #     print(in_line_tensor[:, i, :].unsqueeze(0))

    print("-----Training-----")
    hidden_size = 128
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = RNN(
        input_size=n_letters, hidden_size=hidden_size, output_size=n_letters
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0003
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_iters = 1000
    print_every = 10
    # plot_every = 3

    for it in range(1, n_iters + 1):
        (
            cat,
            line,
            cat_tensor,
            input_line_tensor,
            target_line_tensor,
        ) = random_training_example()

        output, loss = train(
            model,
            cat_tensor,
            input_line_tensor,
            target_line_tensor,
            criterion,
            optimizer,
            device,
        )

        if it % print_every == 0:
            print(f"Iteration: {it}, Loss: {loss}")

