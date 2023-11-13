## Training loop
import torch
from .util import *
def train_ar(model, lr, train_input, batch_size, epos):

    # model = LeNetMNIST(nb_classes=256)
    # if torch.cuda.is_available():
    #     model.cuda()
    optimizer = torch.optim.SGD(params=model.parameters(),lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu, std = train_input.mean(), train_input.std()
    # batch_size = 256
    # epos = 200
    mu, std = train_input.mean(), train_input.std()
    model.train()
    for i in range(epos):
        running_loss = 0
        for data in train_input.split(batch_size):
            # Make 1d sequences from the images
            sequences = tensor2seq(data.to(device) )
            nb, len = sequences.size(0), sequences.size(1)
            # Select a random index in each sequence, this is our targets
            idx = torch.randint(len, (nb, 1), device = device)
            targets = sequences.gather(1, idx).view(-1)
            targets = targets.long()
            # Create masks and values accordingly
            tics = torch.arange(len, device = device).view(1, -1).expand(nb, -1)
            masks = seq2tensor((tics < idx.expand(-1, len)).float())
            values = (data.to(device).float() - mu) / std * masks
            #values = data.float()
            # Make the input, set the mask and values as two channels
            input = torch.cat((masks, values), 1)
            # Compute the loss and make the gradient step
            output = model(input)
            loss = loss_fn(output, targets)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"for {i} epo the loss: {running_loss/batch_size}")
    return model