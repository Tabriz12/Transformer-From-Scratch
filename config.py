
# Model parameters

embedding_dim = 512
d_k_q = 64
d_v = 64
attention_heads = 8
encoder_stack = 6
decoder_stack = 6
batch_size = 16

#Data

dataset = 'wmt16'
subset = 'tr-en'
tokenizer = 'cl100k_base'
voc_size = 100277
pad_token = 0


#training
device = 'cuda'
epoch = 10