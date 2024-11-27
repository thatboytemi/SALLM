import sentencepiece as spm

input_file = 'training_corpus.txt'
model_prefix = 'model_name'
vocab_size = 50000

spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size
)