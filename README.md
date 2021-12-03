## Listen-Attend-Spell Implementation

This is the PyTorch-based model implementation of the paper ["Listen, Attend and Spell"](https://arxiv.org/abs/1508.01211) by Chan et al., 2015 from Google Brain and Carnegie Mellon University. 

This is an  attention-based encoder-decoder model transcribing speech utterance to text in a character-based manner. It utilizes the pyramidal RNN layer to reduce the length of input utterance and attention mechanism to decode information captured by the encoder.

---

#### Highlights of the model implementation

1. **Pyramidal LSTM Layer**
    The utterance is typically quite long so as to exceed 1000. This make the attention mechanism difficult to focus on the right part of the speech during decoding and slower to converge. To tackle this problem, each pyramial LSTM layer reduce the length of input utterance by half, by concatenating the adjacent two. Essentially, after one layer of pyramial LSTM layer, a batch data of size ```(batch_size, seq_len, feat_size)``` becomes ```(batch_size, seq_len / 2, feat_size * 2)```. If the ```seq_len``` is odd, we just chop off the last one.

2. **Locked Dropout**
    We self-implement and insert locked dropout layer in between pyramidal lstm layers. Locked dropout is the way apply the same dropout mask to every time step. This is an efficient way to enhance the generalizability of the encoder. The whole encoder's baseline architecture is therefore ```[lstm -> plstm -> locked-dropout -> plstm -> locked-dropout -> plstm]```
    
3. **Attention Mechanism**
    The model utilizes the attention mechanism to help the decoder to focus on the right part of the speech utterance during decoding. There are many ways of implementin the attention. In this implementation, we use linear transformation to produce ```attention_key``` and ```attention_value``` to be coupled with ```query``` during each timestamp's decoding.

4. **Teacher Forcing**
    It's difficult at early stage for the model to learn because if the decoding at current tiemstamp ```t``` is wrong, then this wrong character's embedding would be feed into ```t+1``` timestamp's decoding, making it even harder to get it right. To tackle this problem, we utilize teacher forcing techniques. Essentially, with a high probability (90% initially), the embedding of `y_{t-1}` to be fed into the decoding process for ```y_t``` would be the ground truth regardless of what the model predicts on last timestamp. As the training process goes, we could gradually decrease the teacher forcing rate and let the model rely wholly on itself. 

5. **Beam Search**
    To fully explore the possible decoding path, we implement beam search in the implementation. However, since it's pretty slow once the beam widthg get bigger, we only used it during validation and inference, and greedy search is applied in the training epochs.

---

#### Where to get the data?

Since the data is too big to be put on the github, we packaged the data source and upload it to google drive for download ([link](https://drive.google.com/file/d/19EPsCrQwdvPoezw7UV_c47Qykyij8T1s/view?usp=sharing)). Please unzip the file and put the files on the ```data/``` folder.

---

#### How to run this model?

1. Check and install dependent packages.
    ```
    pip install -r requirements.txt
    ```
2. ```cd``` to the ```src``` folder
3. (Optionally) Change the hyperparameters as needed in the ```main.py``` script
4. To train, run
    ```
    python3 main.py train
    ```
5. To inference, run
    ```
    python3 main.py infer
    ```
---

#### Hyperparameter Setting

We adopt the baseline architecture setting as described in the paper.

***Decoder***:
+ 1 layer of normal LSTM followed by 3 layers of pyramial LSTM of ```hidden dim=256```. This reduces the input utterance length by a factor of 8.

***Attention***:
+ linear transformation of ```key_value dim=128```

***Decoder***:
+ 2 layers of LSTM cells of ```hidden dim=512```
+ ```character embedding dim=128```

***Optimization***:
+ ```batchsize=64```
+ ```Adam``` optimzer of `lr=0.001`
+ `ReduceLROnPlateau` scheduler of reduce factor ```0.75``` with ```patience=2```, start to step only after first ```10``` epochs
+ train for ```40``` epochs
+ Teacher Forcing rate remains ```95%``` for the first ```10``` epochs and then gradually decrease to ```70%``` by linear interpolation.

The model should be able to reach average Levenshtein distance below 30 on validation set after 40 epochs training.

---

Feel free to email me at yukunj@cs.cmu.edu for questions or discussion for this implementation.
