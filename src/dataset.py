import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from utils import letter2index, index2letter, transform_letter_to_index

# Load the dataset during initialization of the Dataset class to save space
class MyDataset(Dataset):
    def __init__(self, X_path, Y_path=None):
        self.X = np.load(X_path, allow_pickle=True, encoding='bytes')["data"]
        self.Y = None
        if Y_path:
            # for every target sentence y, we append <eos> at the end
            self.Y = [transform_letter_to_index(list(sentence)) + [letter2index['<eos>']] for sentence in \
                      np.load(Y_path, allow_pickle=True,encoding='bytes')['data']]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        # For testing set, return only x
        if self.Y is None:
            return torch.as_tensor(self.X[index].astype(np.float32))
        # For training and validation set, return x and y
        else:
            return torch.as_tensor(self.X[index].astype(np.float32)), torch.as_tensor(self.Y[index])

def collate_train_val(data):
    """
    Return:
        packed_x: the packed-padded x (training/validation speech data)
        packed_y: the packed-padded y (text labels - transcripts)
        x_lenths: the length of x
        y_lenths: the length of y
    """
    # Stick with "batch_first = True"
    (batch_X, batch_Y) = zip(*data)
    x_lengths = torch.tensor([len(x) for x in batch_X])
    y_lengths = torch.tensor([len(y) for y in batch_Y])
    padded_x = pad_sequence(batch_X, batch_first=True)
    padded_y = pad_sequence(batch_Y, batch_first=True)
    packed_x = pack_padded_sequence(padded_x, x_lengths, batch_first=True, enforce_sorted=False)
    packed_y = pack_padded_sequence(padded_y, y_lengths, batch_first=True, enforce_sorted=False)
    return packed_x, packed_y, x_lengths, y_lengths

def collate_test(data):
    """
    Return:
        packed_x: the packed-padded x (testing speech data)
        x_lengths: the length of x
    """
    batch_X = data
    x_lengths = torch.tensor([len(x) for x in batch_X])
    padded_x = pad_sequence(batch_X, batch_first=True)
    packed_x = pack_padded_sequence(padded_x, x_lengths, batch_first=True, enforce_sorted=False)
    return packed_x, x_lengths

if __name__ == "__main__":
    # Local Tesing
    train_path = "../data/train.npz"
    train_scripts_path = "../data/train_transcripts.npz"
    print("local testing...")
    print("loading from train_path=\"{}\" and train_scripts_path=\"{}\"".format(train_path, train_scripts_path))
    batch_size = 128
    num_workers = 2
    train_dataset = MyDataset(train_path, train_scripts_path)
    train_args = {'batch_size' : batch_size, 'shuffle' : True, \
                  'num_workers' : num_workers, 'collate_fn' : collate_train_val}
    train_loader = DataLoader(train_dataset, **train_args)
    print("Finish loading train loader!")

