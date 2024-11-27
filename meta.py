import pickle
meta = {
    'vocab_size': 50000,
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)