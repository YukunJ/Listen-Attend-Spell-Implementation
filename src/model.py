import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
from torch.utils import data
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from attention import Attention
from utils import LETTER_LIST, letter2index, index2letter

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

class MyLockedDropout(nn.Module):
    """
    LockedDropout applies the same dropout mask to every time step
    """
    
    def forward(self, x, dropout=0.2):
        """
        @param: x[tensor] : the tensor we want to apply Locked Dropout upon
                            should be of shape (batch_size, seq_len, frequency)
        """
        # if it's not in the training mode or 0 dropout required, directly return
        if (not self.training) or (dropout == 0):
            return x
        x, x_lengths = pad_packed_sequence(x, batch_first=True)
        # change x to (seq_len, batch_size, frequency)
        x = torch.permute(x, (1, 0, 2))
        
        # construct a mask of size (frequency) that will be applied to every timestamp
        # keep the auxiliary axis for broadcasting
        mask = Variable(x.data.new(1, x.shape[1], x.shape[2]).bernoulli(p=1-dropout) / (1-dropout), requires_grad=False)
        
        # masking off samely for every timestamp and permute back the dimension
        x = torch.permute(mask * x, (1, 0, 2))
        x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        return x


class pBLSTM(nn.Module):
    """
    Pyramidal BiLSTM
    Read paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad back the packed input
    2. Truncate the input length dimension by concatenating feature dimension
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    """
    
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True, num_layers=1)

    def forward(self, x):
        x, x_lengths = pad_packed_sequence(x, batch_first=True)
        B, seq_len, freq_len = x.shape
        if seq_len % 2 == 1:
            # if odd sequence length, chop off the last timestamp
            # single timestamp doesn't matter too much
            x = x[:, :-1, :]
        x_shrinked = x.reshape(B, seq_len // 2, freq_len * 2)
        seq_len /= 2
        packed_x = pack_padded_sequence(x_shrinked, x_lengths // 2, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.blstm(packed_x)
        return output

class Encoder(nn.Module):
    """
    Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.
    """
    
    def __init__(self, input_dim, encoder_hidden_dim, key_value_size=128):
        super(Encoder, self).__init__()
        # The first LSTM layer at the bottom
        self.lstm = nn.LSTM(input_dim, encoder_hidden_dim, bidirectional=True, batch_first=True, num_layers=1)

        # Define the blocks of pBLSTMs to reduce the length of input sequence
        # Insert LockedDropout between plstm layers
        self.pBLSTMs = nn.Sequential(
            pBLSTM(encoder_hidden_dim * 4, encoder_hidden_dim),
            MyLockedDropout(),
            pBLSTM(encoder_hidden_dim * 4, encoder_hidden_dim),
            MyLockedDropout(),
            pBLSTM(encoder_hidden_dim * 4, encoder_hidden_dim),
        )
        self.shrinkage = 3
        # The linear transformations for producing Key and Value for attention
        self.key_network = nn.Linear(2 * encoder_hidden_dim, key_value_size)
        self.value_network = nn.Linear(2 * encoder_hidden_dim, key_value_size)

    def forward(self, x, x_len):
        """
        1. Pack input and pass it through the first LSTM layer (no truncation)
        2. Pass it through the pyramidal LSTM layer
        4. Get output Key, Value, and truncated input lens
        """
        output, (h, c) = self.lstm(x)
        output = self.pBLSTMs(output)
        output, x_lengths = pad_packed_sequence(output, batch_first=True)
        attention_key, attention_value = self.key_network(output), self.value_network(output)
        return attention_key, attention_value, x_len // (2 ** self.shrinkage)

class Decoder(nn.Module):
    """
    Each forward call of decoder deals with just one time step.
    Thus we use LSTMCell instead of LSTM here.
    The output from the last LSTMCell is used as a query for calculating attention.
    Teacher forcing is incorporated for improving the learning at early stage of training.
    """
    
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.decoder_hidden_dim = decoder_hidden_dim
        self.embed_dim = embed_dim
        self.key_value_size = key_value_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # The number of cells is defined based on the paper, i.e. 2 Layers of decoding
        self.lstm1 = nn.LSTMCell(key_value_size+embed_dim,decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(decoder_hidden_dim, key_value_size)
        self.attention = Attention()
        self.vocab_size = vocab_size
        self.character_prob = nn.Linear(2 * key_value_size, vocab_size)

    def forward(self, key, value, encoder_len, y=None, mode='train', TF_rate=0.95):
        """
        Args:
            key :(B, T, d_k) - attention key output of the Encoder
            value: (B, T, d_v) - attention value output of the Encoder
            y: (B, text_len) - Batch input of text with text_length
            mode: train or eval mode for teacher forcing
        Return:
            predictions: the character perdiction probability
        """
        B, key_seq_max_len, key_value_size = key.shape
        
        if mode == 'train':
            max_len =  y.shape[1]
            char_embeddings = self.embedding(y) # for teacher-forcing, feed in ground truth
        else:
            max_len = 500
            
        mask = Variable(torch.zeros(B, key_seq_max_len), requires_grad=False).to(device)
        for k in range(B):
            mask[k][:encoder_len[k]] = 1
        predictions = []
        prediction = torch.full((B,), fill_value=letter2index['<sos>'], device=device)
        hidden_states = [None, None]
        context = torch.zeros((B, self.key_value_size), device=device)
        attention_plot = [] # this is for debugging and visualization

        for i in range(max_len):
            """
            For first timestamp, always feed in <sos> token embedding
            For the rest timestamp, roll a dice to decide if use teacher-forcing
            """
            if mode == 'train':
                if i == 0:
                    char_embed = self.embedding(prediction)
                else:
                    if random.random() <= TF_rate:
                        char_embed = char_embeddings[:, i-1, :]
                    else:
                        char_embed = self.embedding(torch.argmax(prediction, dim=1))
            else:
                if i == 0:
                    char_embed = self.embedding(prediction)
                else:
                    char_embed = self.embedding(torch.argmax(prediction, dim=1))
                    
            # decoding through two layers to get the attention query
            y_context = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(y_context, hidden_states[0])
            hidden_states[1] = self.lstm2(hidden_states[0][0], hidden_states[1])
            query = hidden_states[1][0]
            
            # Compute attention from the output of the second LSTM Cell
            # We store the first attention of this batch for debugging and visualization
            context, attention = self.attention(query, key, value, mask)
            attention_plot.append(attention[0].detach().cpu())
            
            # Concatenate query and context and apply linear layer to get prob distribution
            output_context = torch.cat([query, context], dim=1)
            prediction = self.character_prob(output_context)
            predictions.append(prediction.unsqueeze(dim=1))
        
        # Concatenate the attention and predictions to return
        attentions = torch.stack(attention_plot, dim=0)
        predictions = torch.cat(predictions, dim=1)
        return predictions, attentions
        
class Seq2Seq(nn.Module):
    """
    Model Wrapper
    end-to-end sequence to sequence model comprising of Encoder and Decoder.
    """
    
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim, key_value_size)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim, key_value_size)

    def forward(self, x, x_len, y=None, mode='train', TF_rate=0.95):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions = self.decoder(key, value, encoder_len, y=y, mode=mode, TF_rate=TF_rate)
        return predictions

if __name__ == "__main__":
    print("Local Testing...")
    input_dim           = 40
    vocab_size          = len(LETTER_LIST)
    encoder_hidden_dim  = 256
    decoder_hidden_dim  = 512
    embed_dim           = 128
    key_value_size      = 128
    TF_rate             = 0.95
    model = Seq2Seq(input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, key_value_size)
    print("printing model architecture...")
    print(model)
    print("Success")
