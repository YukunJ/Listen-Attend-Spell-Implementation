import sys
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torch.utils.data import Dataset, DataLoader
from utils import LETTER_LIST, letter2index, index2letter, transform_letter_to_index
from dataset import MyDataset, collate_train_val, collate_test
from search import greedy_search, beam_search
from model import Seq2Seq
from trainer import train, val, inference
warnings.filterwarnings('ignore')

def train_N_epoch(num_epochs, model, train_loader, valid_loader, optimizer, scheduler=None, store_path="../checkpoint", beam_width=8):
    best_loss = float('inf')
    for i in range(1, num_epochs+1):
        if i <= 10: # warm up for the first 10 epochs
            TF_rate = 0.95
        else:
            # ideally, when train up to 40 epochs,
            # the teacher forcing rate gradually drops to 70%
            TF_rate = 0.95 * 70 / ((i-10) + 70)
        for g in optimizer.param_groups:
            # retrieve the current learning rate
            curr_lr = g['lr']
        print("In epoch {} using lr_rate {:.5f} TF rate {:.1f}%".format(i, curr_lr, 100 * TF_rate))
        cross_loss, perp_loss, dist = train(model, train_loader, criterion, optimizer, 'train', TF_rate)
        cross_loss, perp_loss, dist = val(model, valid_loader, criterion, beam_width)
        if i >= 10 and scheduler is not None:
            # do not step for the first 10 epochs' warmup
            scheduler.step(perp_loss)
            
        # save checkpoint every 5 epochs
        if (i % 5 == 1):
            print("saving a model checkpoint at epoch {}".format(i))
            torch.save(model.state_dict(), store_path + "/model_dict_" + str(i) + ".pt")
        if perp_loss < best_loss:
            print("best model updates at epoch {}".format(i))
            best_loss = perp_loss
            torch.save(model.state_dict(), store_path + "/model_dict_best.pt")

if __name__ == "__main__":

    # feel free to change these hyperparameters for tuning
    # the provided below is the baseline architecture setting in the LAS paper
    training_epochs     = 40
    input_dim           = 40
    vocab_size          = len(LETTER_LIST)
    encoder_hidden_dim  = 256
    decoder_hidden_dim  = 512
    embed_dim           = 128
    key_value_size      = 128
    batch_size          = 64
    num_workers         = 4 if torch.cuda.is_available() else 1
    beam_width          = 8
    store_path = "../checkpoint"
    output_path = "../output"
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(device, sys.version)
    criterion = nn.CrossEntropyLoss(reduction='none')
    model = Seq2Seq(input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, key_value_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=2, verbose=True)
    print("Finish Model initialization")
    if sys.argv[1] == 'train':
        print("Training Mode")
        train_dataset = MyDataset("../data/train.npz", "../data/train_transcripts.npz")
        valid_dataset = MyDataset("../data/dev.npz", "../data/dev_transcripts.npz")
        train_args = {'batch_size' : batch_size, 'shuffle' : True, \
                      'num_workers' : num_workers, 'collate_fn' : collate_train_val}
        valid_args = {'batch_size' : batch_size, 'shuffle' : False, \
                      'num_workers' : num_workers, 'collate_fn' : collate_train_val}
        train_loader = DataLoader(train_dataset, **train_args)
        valid_loader = DataLoader(valid_dataset, **valid_args)
        print("Finish loading train/valid dataset")
        model.to(device)
        train_N_epoch(training_epochs, model, train_loader, valid_loader, optimizer, scheduler, store_path, beam_width)
        
    elif sys.argv[1] == 'infer':
        print("Inference Mode")
        test_dataset = MyDataset("../data/test.npz")
        test_args = {'batch_size' : batch_size, 'shuffle' : False, \
              'num_workers' : num_workers, 'collate_fn' : collate_test}
        test_loader = DataLoader(test_dataset, **test_args)
        print("Finish loading test dataset")
        model.load_state_dict(torch.load(store_path + "/model_dict_best.pt"))
        model.to(device)
        inference(model, test_loader, output_path="../output", beam_width=beam_width)

