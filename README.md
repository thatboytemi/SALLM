# Multilingual Language Model For South African Languages

## Author: Temiloluwa Aina

### Tokenization

The data is stored in jsonl files as json objects with the 'text' field containing the training text. The tokenizer requires one text file with all the data. To get the data into the required format, run the extractor.py file.
To tokenize the input corpus, we trained a SentencePiece Unigram Language Model. The code for this can be found in the tokenizer.py file.

### NanoGPT

The nanoGPT model expects the training tokens to be placed in one file (called train.bin) and the validation tokens to be placed in another file (called val.bin). The code to to tokenize the corpus using trained tokenizer is in the prepare_nanogpt file.

### Modded-NanoGPT

The modded nanoGPT model expects the training and validation tokens to be placed in separate files of size 10^8 tokens. The code to to tokenize the corpus using trained tokenizer is in the prepare_modded file.

### Experiments

We conducted preliminary experiments to identify which GPT implementation to use. Below are the training and validation loss curves of various NanoGPT and Modded-NanoGPT models trained. In the NanoGPT model, we varied the base learning rate by increasing and decreasing it by a factor of 10 and for the Modded-Nano model we did the same but only for the learning rate of the Muon optimizer.

![alt text](<W&B Chart 11_27_2024, 12_24_25 PM.png>)

![alt text](<W&B Chart 11_27_2024, 12_24_56 PM.png>)

The training times for each model are shown below.

|        Model         | Training Time |
| -------------------- | ------------- |
|Modded: muon lr= 0.02 |    56m 30s    | 
|Modded: muon lr= 0.002 |    56m 54s    | 
|Modded: muon lr= 0.0002 |    57m 14s    | 
|Nano: lr= 6e-3 |   2h 20m 5s    | 
|Nano: lr= 6e-2 |    2h 16m 43s    | 
|Nano: lr= 6e-4 |    2h 19m 34s    | 
|Nano: lr= 6e-3 Dropout= 0.05 |    2h 21m 59s    | 
|Nano: lr= 6e-3 Dropout= 0.1 |    2h 22m 32s    | 




The results show that the Modded-NanoGPT model would be the best implementation to use because it achieved the lowest validation error in the shortest amount of time. The results show that a higher learning rate for the Muon optimizer (of 0.02) may yield the best results. 


### Model code

The modified versions of the NanoGPT and Modded-NanoGPT training files can be found nanotrain.py and moddedtrain.py.
For early stopping, added a counter called 'patience' that tracks the number of evaluations since the best validation loss was obtained. If the patience counter exceeds 10, training is halted.
I modified the NanoGPT code to fetch training batches sequentially rather than sample them randomly like in the original code.
The nanoGPT file expects a meta.pkl file that contains the vocab size of the model. The code to create this file is contained in meta.py.

# Requirements

pip install numpy transformers datasets wandb 
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade 
This specific version of torch is needed to enable torch.compile()
I made use of Python 3.12 when training.

# Running models

Sample run command: torchrun --standalone --nproc_per_node=4 moddedtrain.py

# Models

The best Modded-Nano model is stored in 'Best Modded' and the best NanoGPT model is stored in 'Best Nano' on the workstation.
 