import torch
import numpy as np
from utils import letter2index, index2letter

def greedy_search(index_lst, stopping=[letter2index['<pad>'], letter2index['<eos>'], letter2index['<sos>']]):
    """
    Greedy search
    @param: index_lst of shape (Batch_size, seq_len) after taking argmax on vocab_size dimension
    """
    translations = []
    for sent_index in index_lst:
        curr = []
        for i in sent_index:
            if i in stopping:
                break
            curr.append(index2letter[i])
        translations.append("".join(curr))
    return translations

def beam_search(raw_logits, beam_width: int=8):
    """
    Beam search
    @param: raw_logits of shape (Batch_size, seq_len, vocab_size)
    """
    
    logits = np.log(raw_logits)
    batch_size, seq_len, vocab_size = logits.shape
    output = []
    for b in range(batch_size): # iterate the whole batch, essentially decoding one by one
        L = logits[b]
        finished = []
        # 1. Init the beam, with log(p=1) = 0
        candidates = [("<sos>", 0.0)]

        # 2. For each time stamp, iterate through the previous remaining <beam_width>
        #    and expand to every current timestamp's vocab_size probability
        #    and filter out the size to <beam_width>, ready for next iteration
        for t in range(seq_len):
            next_candidates = []
            for prev_str, prev_score in candidates:
                for v in range(vocab_size):
                    next_str = prev_str + index2letter[v]
                    next_score = prev_score + L[t][v]
                    next_candidates.append((next_str, next_score))

            # got all the candidates for current timestamp, trim them to <beam_width> by sorting
            almost_candidates = sorted(next_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # trim all the candidates ending with <eos> or <pad>, they are terminated already
            candidates = []
            for curr_str, curr_score in almost_candidates:
                if curr_str.endswith("<eos>") or curr_str.endswith("<pad>") or curr_str.endswith("<sos>"):
                    finished.append((curr_str, curr_score))
                else:
                    candidates.append((curr_str, curr_score))
        
        # 3. Got all the available probability so far, reorder them and take the most likely one
        finished.extend(candidates)
        finished.sort(key=lambda x: x[1], reverse=True)
        output.append(finished[0][0].replace('<sos>', '').replace('<eos>', '').replace('<pad>', ''))

    return output

if __name__ == "__main__":
    print("Local Testing...")
    greedy_search(np.zeros((32,200)))
    beam_search(np.zeros((32, 200, 32)))
    print("Success")
