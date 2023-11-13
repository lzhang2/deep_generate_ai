import torch
from .util import *
def inference(model, nb, shape, mu, std):
    #shape = train_input.shape[1:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generated = torch.zeros((nb,) + shape,device = device, dtype = torch.int64)
    sequences = tensor2seq(generated)
    tics = torch.arange(sequences.size(1), device = device).view(1, -1).expand(nb, -1)
    for t in range(sequences.size(1)):
        masks = seq2tensor((tics < t).float())
        values = (seq2tensor(sequences).float() - mu) / std * masks
        input = torch.cat((masks, values), 1)
        output = model(input)
        dist = torch.distributions.categorical.Categorical(logits = output)
        sequences[:, t] = dist.sample()
    return sequences

def inference_with_prior(model, nb, shape, mu, std, prior):
    #shape = train_input.shape[1:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generated = torch.zeros((nb,) + shape,device = device, dtype = torch.int64)
    _, nrow, _ = prior.shape
    generated[:,:,:nrow,:] = prior
    sequences = tensor2seq(generated)
    tics = torch.arange(sequences.size(1), device = device).view(1, -1).expand(nb, -1)
    for t in range(nrow*28,sequences.size(1)):
        masks = seq2tensor((tics < t).float())
        values = (seq2tensor(sequences).float() - mu) / std * masks
        input = torch.cat((masks, values), 1)
        output = model(input)
        dist = torch.distributions.categorical.Categorical(logits = output)
        sequences[:, t] = dist.sample()
    return sequences