
# Model parameters

embedding_dim = 512
d_k_q = 64
d_v = 64
attention_heads = 8
encoder_stack = 6
decoder_stack = 6
batch_size = 12

#Data

dataset = 'wmt16'
subset = 'tr-en'
tokenizer = 'cl100k_base'
voc_size = 100277


#training
device = 'cuda'
epoch = 10