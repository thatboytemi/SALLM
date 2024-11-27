"""
File containing the code for tokenization.
Encoding using SentencePiece unigram model.
Training and validation sets saved to train_*.bin and valid_*.bin

Expected input is a text file containing the input corpus.
"""
import numpy as np
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file = "test_model.model")

#Tokenize training set
train_ids = []
with open('/scratch/anxtem001/train.txt', 'r', encoding="utf-8") as file:
    for line in file:
        train_ids.extend(sp.tokenize(line))
print("Tokenized train")
#Tokenize validation set
val_ids = []
with open('/scratch/anxtem001/val.txt', 'r', encoding="utf-8") as file:
    for line in file:
        val_ids.extend(sp.tokenize(line))
print("Tokenized val")
#Divide training set into blocks of size 1024 tokens
train_chunks = []
for i in range(0, len(train_ids), 1024):
    train_chunks.append(train_ids[i:i + 1024])
del train_ids
#Divide validation set into blocks of size 1024 tokens
val_chunks = []
for i in range(0, len(val_ids), 1024):
    val_chunks.append(val_ids[i:i + 1024])
del val_ids
#Randomise the order of the training and validation blocks
np.random.seed(42)
np.random.shuffle(train_chunks)
np.random.shuffle(val_chunks)

print("Shuffled data")
#Flatten the 2D arrays into 1D arrays
x = []
y = []
for array in train_chunks:
    x += array
del train_chunks
for array in val_chunks:
    y += array
del val_chunks

#convert arrays to numpy arrays for efficient storage
train_ids = np.array(x, dtype=np.uint16)
val_ids = np.array(y, dtype=np.uint16)
del x
del y
#store training and validation sets in files of size 10^8 tokens
for i in range(0, train_ids.shape[0], 10**8):
    with open(f"/scratch/anxtem001/data/train_{i}.bin", "wb") as f:
        train_ids[i:i+10**8].tofile(f)
del train_ids
print("Train written")
for i in range(0, val_ids.shape[0], 10**8):
    with open(f"/scratch/anxtem001/data/valid_{i}.bin", "wb") as f:
        val_ids[i:i+10**8].tofile(f)
del val_ids
print("Val written")