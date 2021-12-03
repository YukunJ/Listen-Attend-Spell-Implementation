import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import Levenshtein
from tqdm import tqdm
from torch.utils import data
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from attention import Attention
from utils import LETTER_LIST, letter2index, index2letter
from search import greedy_search, beam_search
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def train(model, train_loader, criterion, optimizer, mode='train', TF_rate=0.95):
    model.train()
    running_entropy_loss = 0.0
    running_perx_loss = 0.0
    running_dist = 0.0
    for i, (packed_x, packed_y, x_lengths, y_lengths) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        x, y, x_len, y_len = packed_x.to(device), packed_y.to(device), x_lengths.to(device), y_lengths.to(device)
        y, _ = pad_packed_sequence(y, batch_first=True)
        predictions, attentions = model(x, x_len, y, mode='train', TF_rate=TF_rate)
        b, seq_max, _ = predictions.shape
        # generate a mask to cross-out those padding's KL-divergence
        mask = Variable(torch.zeros(b, seq_max), requires_grad=False).to(device)
        for k in range(b):
            mask[k][:y_len[k]] = 1
        loss = criterion(predictions.view(-1, predictions.shape[2]), y.view(-1))
        avg_masked_loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)
        running_entropy_loss += float(avg_masked_loss.item())
        running_perx_loss += float(torch.exp(avg_masked_loss).item())
        avg_masked_loss.backward()
        # visualize graident distribution and attention plot for debugging purpose
        # if you are using jupyter notebook, you can comment out the below to see plots
        # if (i == 0):
        #     plot_grad_flow(model.named_parameters(), i, i)
        #     plot_attention(attentions)
        
        # clipping to avoid graident exploding
        clipping_value = 1.0
        torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
        optimizer.step()
        # translation for Lev Distance, use greedy search during training
        pred_strs = greedy_search(torch.argmax(predictions, dim=-1).detach().cpu().numpy())
        ans_strs = greedy_search(y.detach().cpu().numpy())
        running_dist += np.mean([Levenshtein.distance(pred, ans) for pred, ans in zip(pred_strs, ans_strs)])
        # clear cuda cache for memory issue
        del x, y, x_len, y_len
        torch.cuda.empty_cache()
    
    running_entropy_loss /= (i+1)
    running_perx_loss /= (i+1)
    running_dist /= (i+1)
    print("Train Result: Cross Entropy Loss : {:.3f} and Perplex Loss : {:.3f} and Lev Dist : {:.3f}".format(running_entropy_loss, running_perx_loss, running_dist))
    return running_entropy_loss, running_perx_loss, running_dist
        
def val(model, valid_loader, criterion, beam_width=8):
    model.eval()
    running_entropy_loss = 0.0
    running_perx_loss = 0.0
    running_dist = 0.0
    with torch.no_grad():
        for i, (packed_x, packed_y, x_lengths, y_lengths) in enumerate(tqdm(valid_loader)):
            x, y, x_len, y_len = packed_x.to(device), packed_y.to(device), x_lengths.to(device), y_lengths.to(device)
            y, _ = pad_packed_sequence(y, batch_first=True)
            # one predictions is ending-controlled for loss computation
            # another is free-generation util max-len reached
            predictions, attentions = model(x, x_len, y, mode='eval')
            predictions_loss, attentions = model(x, x_len, y, mode='train')
            # generate a mask to cross-out those padding's KL-divergence
            b, seq_max, _ = predictions_loss.shape
            mask = Variable(torch.zeros(b, seq_max), requires_grad=False).to(device)
            for k in range(b):
                mask[k][:y_len[k]] = 1
            loss = criterion(predictions_loss.view(-1, predictions_loss.shape[2]), y.view(-1))
            avg_masked_loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)
            running_entropy_loss += float(avg_masked_loss.item())
            running_perx_loss += float(torch.exp(avg_masked_loss).item())
            # translation for Lev Distance, using beam search in validation
            beam_pred_strs = beam_search(F.softmax(predictions, dim=-1).detach().cpu().numpy(), beam_width)
            ans_strs = greedy_search(y.detach().cpu().numpy())
            running_dist += np.mean([Levenshtein.distance(pred, ans) for pred, ans in zip(beam_pred_strs, ans_strs)])
            del x, y, x_len, y_len
            torch.cuda.empty_cache()
    
    running_entropy_loss /= (i+1)
    running_perx_loss /= (i+1)
    running_dist /= (i+1)
    print("Valid: Cross Entropy Loss : {:.3f} and Perplex Loss : {:.3f} and Lev Dist(beam width={}) : {:.3f}".format(running_entropy_loss, running_perx_loss, beam_width, running_dist))
    return running_entropy_loss, running_perx_loss, running_dist

def inference(model, test_loader, output_path="../output", beam_width=8):
    """
    inference
    """
    
    def test(model, test_loader, beam_width):
        """
        Generate testing string transaltion for the test dataset
        """
        model.eval()
        str_predictions = []
        with torch.no_grad():
            for i, (packed_x, x_lengths) in enumerate(tqdm(test_loader)):
                x, x_len = packed_x.to(device), x_lengths.to(device)
                predictions, attentions = model(x, x_len, y=None, mode='eval')
                # You can choose to use greedy search(more efficient) or beam search(more exploratory) 
                # pred_strs = greedy_search(torch.argmax(predictions, dim=-1).detach().cpu().numpy())
                pred_strs = beam_search(F.softmax(predictions, dim=-1).detach().cpu().numpy(), beam_width)
                str_predictions.extend(pred_strs)
                del x, x_len
                torch.cuda.empty_cache()
        return str_predictions
        
    def output(predictions, output_path):
        """
        Output the inference result to the proper csv file with column header for submission

        @param:
            inferences [List]  : inferenced state label for test dataset
        """
        df = pd.DataFrame(predictions, columns=["label"])
        df.reset_index(inplace=True)
        df = df.rename(columns = {'index':'id'})
        df.reset_index(drop=True, inplace=True)
        df.to_csv(output_path + "/submission.csv", index=False)
        print("Finish storage of submission file, check if out in your folder {}!".format(output_path + "/submission.csv"))
        
    str_predictions = test(model, test_loader, beam_width)
    output(str_predictions, output_path)
    
def plot_grad_flow(named_parameters, epoch, batch):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as
          "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.7, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="r")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    ax = plt.gca()
    ax.tick_params(axis='x',which='major',labelsize=6)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.0001, top=0.01) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.show()
    
def plot_attention(attention):
    # utility function for debugging
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()