import torch
import torch.nn.functional as F
from app.model import Linear, Tanh


async def load_params():
    """Loads pre-trained parameters of the MLP neural network and a lookup dict.
    Returns `C`, `layers`, `itos`"""

    # Read in the trained parameters and the lookup dict.
    # All weights and biases of the Linear layers have been
    # BatchNorm folded for improved inference performance.
    params = torch.load('app/6_layer_trained_params.pt')
    itos = torch.load('app/lookup.pt')

    # Initialize the neural network
    block_size = 3
    n_embeddings = 10
    vocab_size = 27
    n_hidden = 100

    g = torch.Generator().manual_seed(2147483647)

    layers = [Linear(block_size * n_embeddings, n_hidden), Tanh(),
              Linear(n_hidden, n_hidden),                  Tanh(),
              Linear(n_hidden, n_hidden),                  Tanh(),
              Linear(n_hidden, n_hidden),                  Tanh(),
              Linear(n_hidden, n_hidden),                  Tanh(),
              Linear(n_hidden, vocab_size)]

    # Load the trained parameters into the neural network
    C = params[0]

    for i in range(0, len(params[1:]), 2):
        layers[i].weight = params[i + 1]
        layers[i].bias = params[i + 2]

    return C, layers, itos


def generate_names(C, layers, itos, number, seed):
    """Generates names by doing a forward pass through the network to predict each next character.
    The context size is 3.

    Returns a list of strings `names`"""


    names = []
    block_size = 3

    g = torch.Generator().manual_seed(seed)

    for i in range(number):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            x = emb.view(emb.shape[0], -1)
            for layer in layers:
                x = layer(x)
            probs = F.softmax(x, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            out.append(ix)
            context = context[1:] + [ix]
            if ix == 0:
                break
        names.append(''.join(itos[i] for i in out))
    return names
