import sentencepiece as spm

input_file = '/scratch/anxtem001/extracted_data.txt'
model_prefix = 'test_model'
vocab_size = 50000

spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    train_extremely_large_corpus=True, 
    input_sentence_size=7000000 #dataset is too large so have to sample sentences to train tokenizer on. Can possibly increase this value
)
