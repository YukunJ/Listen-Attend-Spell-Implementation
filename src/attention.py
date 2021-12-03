import torch
import torch.nn as nn
import numpy as np

class Attention(nn.Module):
    '''
    Dot Attention is calculated using key and value from encoder and query from decoder.
    '''
    def __init__(self):
        super(Attention, self).__init__()
        # Optional: dropout

    def forward(self, query, key, value, mask):
        """
        input:
            key: (batch_size, seq_len, d_k)
            value: (batch_size, seq_len, d_v)
            query: (batch_size, d_q)
        return:
            context: (batch_size, key_val_dim)
        
        """
        b, seq_len, key_val_dim = key.shape
        energy = torch.bmm(key, query.unsqueeze(dim=2)).squeeze(dim=2)
        # masking off the padded index to zero out their attention energy
        attention = nn.functional.normalize(mask * nn.functional.softmax(energy, dim=1), p=1, dim=1)
        context = torch.bmm(attention.unsqueeze(dim=1), value).squeeze(dim=1)
        return context, attention

if __name__ == "__main__":
    print("Local Testing...")
    myAttention = Attention()
    print("Success")
