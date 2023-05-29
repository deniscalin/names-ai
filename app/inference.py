import torch
import torch.nn.functional as F


async def load_params():
    """Loads pre-trained parameters of the MLP neural network and a lookup dict.
    Returns two dictionaries `params`, `itos`"""
    params = torch.load('app/params.pt')
    itos = torch.load('app/lookups.pt')
    return params, itos


def generate_names(params, itos, number, seed):
    """Generates names by doing a forward pass through the network to predict each next character.
    The context size is 3.

    Returns a list of strings `names`"""
    C = params['C']
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    names = []
    block_size = 3

    g = torch.Generator().manual_seed(seed)

    for _ in range(number):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, 1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g)
            out.append(ix.item())
            context = context[1:] + [ix.item()]
            if ix == 0:
                break
        names.append(''.join(itos[i] for i in out))
    return names
