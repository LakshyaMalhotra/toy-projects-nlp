import torch
import torch.nn as nn


class Engine:
    def __init__(self, model, optimizer, criterion, device, clip):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.clip = clip

    def train(self, iterator):
        self.model.train()
        epoch_loss = 0.0

        for batch in iterator:
            input_tensor = batch.src
            target_tensor = batch.trg

            # move the tensors to cuda if available
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)

            # zeroing out any previous gradients
            self.optimizer.zero_grad()

            # forward pass
            output = self.model(input_tensor, target_tensor)
            # output shape: [seq_len, batch_size, output_size]

            output_size = output.size(-1)

            # ignoring the <sos> token
            output = output[1:].view(-1, output_size)
            target_tensor = target_tensor[1:].view(-1)

            # calculate loss
            loss = self.criterion(output, target_tensor)

            # backpropagate
            loss.backward()

            # clip the gradients if they exceed beyond certain value
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            # step the optimizer
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def evaluate(self, iterator):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in iterator:
                input_tensor = batch.src
                target_tensor = batch.trg

                # move the tensors to cuda if available
                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)

                # zeroing out any previous gradients
                self.optimizer.zero_grad()

                # forward pass
                output = self.model(input_tensor, target_tensor)
                # output shape: [seq_len, batch_size, output_size]

                output_size = output.size(-1)

                # ignoring the <sos> token
                output = output[1:].view(-1, output_size)
                target_tensor = target_tensor[1:].view(-1)

                # calculate loss
                loss = self.criterion(output, target_tensor)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)
