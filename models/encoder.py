import torch
from torch import nn
from torch.nn import functional as F
import math
import config
import einops


def scaled_dot_product_attention(Key: torch.Tensor, Query: torch.Tensor, Value: torch.Tensor, 
                                 mask: torch.Tensor = None, attending_idx: int = None):

    '''

    mask -> batch_size, seq_len, seq_len
    
    '''

    if attending_idx:
        '''

        fill all with -inf after attending idx i mask
        
        '''
        pass

    mask = einops.repeat(mask, 'b e s -> b h e s', h=config.attention_heads) # adding heads

    attn = (Query @ Key.permute(0,1,3,2))/math.sqrt(Key.size(-1))# batch size, heads, seq_len, seq_len

    attn = F.softmax(attn + mask, dim=-1) @ Value # Batch size, heads, seq_len, dim
    
    return attn



def position_embedding(x: torch.Tensor):

    res = torch.zeros_like(x)

    for batch_id, batch in enumerate(x):
        for pos, token in enumerate(batch):

            if pos % 2 == 0:
                vector = torch.FloatTensor([math.sin(pos/10000**(idx/config.embedding_dim)) for idx in range(config.embedding_dim)])
            
            else:
                vector = torch.FloatTensor([math.cos(pos/10000**(idx/config.embedding_dim)) for idx in range(config.embedding_dim)])
            
            res[batch_id, pos] = vector
    
    assert x.size() == res.size()
    

    return res

class FeedForward(nn.Module):
    def __init__(self, dim_in = config.embedding_dim,
                 dim_mid = config.embedding_dim*4):

        super().__init__()

        self.fc1 = nn.Linear(dim_in, dim_mid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim_mid, dim_in)
        self.layer_norm = nn.LayerNorm(dim_in)
    
    def forward(self, x):

        skip = x

        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        out = self.layer_norm(x+skip)

        return out
    

class MultiHeadAttention(nn.Module):


    def __init__(self,
                embedding_dim = config.embedding_dim,
                d_k_q = config.d_k_q, 
                d_v = config.d_v,
                heads = config.attention_heads,
                attention_f = scaled_dot_product_attention):

        super().__init__()
        self.d_k_q = d_k_q
        self.d_v = d_v
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.attention_f = attention_f

        self.key_mapping = nn.Linear(embedding_dim, d_k_q * heads, bias=False)
        self.query_mapping = nn.Linear(embedding_dim, d_k_q * heads, bias=False)
        self.value_mapping = nn.Linear(embedding_dim, d_v * heads, bias=False)

        self.last_linear = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)


    def forward(self, x: torch.Tensor, x_attention: torch.Tensor):

        '''
        x -> [batch_size, sequence length, embedding_dim ]
        x_attention -> [batch_size, sequence length  ]

        '''

        Key = self.key_mapping(x)
        Query = self.query_mapping(x)
        Value = self.value_mapping(x)

        Key = Key.reshape(x.size(0), x.size(1), self.heads, self.d_k_q) # Batch size, seq_len, heads, dim
        Query = Query.reshape(x.size(0), x.size(1), self.heads, self.d_k_q)
        Value = Value.reshape(x.size(0), x.size(1), self.heads, self.d_v)

        Key = Key.permute(0,2,1,3)  # Batch size, heads, seq_len, dim
        Query = Query.permute(0,2,1,3)
        Value = Value.permute(0,2,1,3)

        attention = self.attention_f(Key, Query, Value, x_attention)

        attention = attention.permute(0,2,1,3)
        
        concatted_attention = attention.reshape(attention.size(0), attention.size(1), -1)

        assert concatted_attention.size() == x.size()

        out = self.last_linear(concatted_attention)

        out = self.layer_norm(out + x)

        return out


    

class Encoder(nn.Module):
    def __init__(self, stack = config.encoder_stack):
        super().__init__()

        self.stack = stack

        self.embedding_layer = nn.Embedding(config.voc_size, config.embedding_dim)

        self.attention_layers = nn.ModuleList([MultiHeadAttention() for _ in range(stack)])
        self.ff_layers = nn.ModuleList([FeedForward() for _ in range(stack)])

    def forward(self, x, x_attention):

        '''
        x -> [batch_size, max num of tokens in the batch ]
        x_attention -> [batch_size, max num of tokens in the batch ]

        '''

        x = self.embedding_layer(x)

        pos_embs = position_embedding(x)

        x = x + pos_embs

        for i in range(self.stack):
            x = self.attention_layers[i](x, x_attention)
            x = self.ff_layers[i](x)
    
        return x

        
    
class Decoder(nn.Module):

    def __init__(self, stack = config.decoder_stack):

        super().__init__()

        self.stack = stack

        self.embedding_layer = nn.Embedding(config.voc_size, config.embedding_dim)

        self.attention_models = nn.ModuleList([MultiHeadAttention() for _ in range(stack)])

        self.endoder_decoder_attentions = nn.ModuleList([MultiHeadAttention() for _ in range(stack)])

        self.feed_forward_model = nn.ModuleList([FeedForward() for _ in range(stack)])
    

    def forward(self, decoder_in, decoder_att, encoder_in, encoder_att):

        decoder_in = self.embedding_layer(x)

        pos_embs = position_embedding(x)

        x = x + pos_embs



 
    