import torch
from torch import nn
from torch.nn import functional as F
import math
import config
import einops


def create_attention_mask(batch):

    for idx, sequence in enumerate(batch):

        max_len = len(sequence)
        seq_len = sum(sequence)

        line = torch.Tensor([
            [0.0] * seq_len + [float('-inf')] * (max_len - seq_len)
        ]).squeeze()

        top = line.unsqueeze(0).repeat(seq_len, 1)
        bottom = torch.full(((max_len-seq_len), max_len), float('-inf'))
        mask_tensor = torch.cat((top, bottom), dim=0).unsqueeze(0)
        
        if not idx: 
            batched_mask = mask_tensor
        else:
            batched_mask = torch.cat((batched_mask,  mask_tensor), dim=0)

    return batched_mask

def create_cross_attention_mask(enc_batch, dec_batch):
    
    for batch_id in range(config.batch_size):
        c = len(enc_batch[batch_id])
        seq_len_e = sum(enc_batch[batch_id])
        seq_len_d = sum(dec_batch[batch_id])
        r = len(dec_batch[batch_id])

        line = torch.Tensor([
            [0.0] * seq_len_e + [float('-inf')] * (c - seq_len_e)
        ]).squeeze()

        top = line.unsqueeze(0).repeat(seq_len_d, 1)
        bottom = torch.full(((r-seq_len_d), c), float('-inf'))
        mask_tensor = torch.cat((top, bottom), dim=0).unsqueeze(0)

        if not batch_id: 
            batched_mask = mask_tensor
        else:
            batched_mask = torch.cat((batched_mask,  mask_tensor), dim=0)
        
    return batched_mask


def scaled_dot_product_attention(Key: torch.Tensor, Query: torch.Tensor, Value: torch.Tensor, 
                                 input_attention_mask: torch.Tensor = None, triange_mask: torch.Tensor = None,
                                 output_attention_mask:torch.Tensor= None):

    '''

    attention_mask, output_attention_mask -> [batch_size, seq_len]
    
    decoder_mask -> [batch_size, seq_len, seq_len]

    Key, Query, Value -> [batch size, heads, seq_len, dim]
    
    '''


    if triange_mask != None:

        if input_attention_mask != None: ## cross attention

            attention_mask = create_cross_attention_mask(input_attention_mask, output_attention_mask)

            print("Cross attention mask: {}".format(attention_mask.size()))


            # + triangle mask

            #decoder_mask = output_attention_mask + decoder_mask

        
        else: ## decoder masked self attention
            attention_mask = create_attention_mask(output_attention_mask)
            attention_mask = attention_mask + triange_mask

    else: ###  self attention (encoder)
        attention_mask = create_attention_mask(input_attention_mask)


    attention_mask = attention_mask.to(config.device)

    attention_mask = einops.repeat(attention_mask, 'b e s -> b h e s', h=config.attention_heads) # adding heads

    attn = (Query @ Key.permute(0,1,3,2))/math.sqrt(Key.size(-1)) # batch size, heads, seq_len, seq_len

    assert attn.size() == attention_mask.size()

    print("Query @ Key size {}".format(attn.size()))

    '''

    attn for self attention -> [batch size, heads, seq_len, seq_len]

    attn for cross attention -> [batch size, heads, seq_len decoder, seq_len encoder]

    @ Value self attention -> [batch size, heads, seq_len, embedding dim] 

    @ Value cross attention -> [batch size, heads, seq_len decoder, embedding dim] 

    '''

    attn = F.softmax(attn + attention_mask, dim=-1) @ Value 
    
    return attn


def scaled_dot_product_attention_old(Key: torch.Tensor, Query: torch.Tensor, Value: torch.Tensor, 
                                 attention_mask: torch.Tensor = None, decoder_mask: torch.Tensor = None,
                                 output_attention_mask:torch.Tensor= None):

    '''

    attention_mask, decoder_mask -> [batch_size, seq_len, seq_len]

    Key, Query, Value -> [batch size, heads, seq_len, dim]
    
    '''
    if decoder_mask:

        if output_attention_mask:

            decoder_mask = output_attention_mask + decoder_mask

        print(attention_mask)

        attention_mask = attention_mask + decoder_mask # adding two masks together

        print(attention_mask)

        s
    

    attention_mask = einops.repeat(attention_mask, 'b e s -> b h e s', h=config.attention_heads) # adding heads

    attn = (Query @ Key.permute(0,1,3,2))/math.sqrt(Key.size(-1))# batch size, heads, seq_len, seq_len

    '''

    attn for self attention -> [batch size, heads, seq_len, seq_len]

    attn for cross attention -> [batch size, heads, seq_len decoder, seq_len encoder]

    @ Value self attention -> [batch size, heads, seq_len, embedding dim] 

    @ Value cross attention -> [batch size, heads, seq_len decoder, embedding dim] 

    '''

    attn = F.softmax(attn + attention_mask, dim=-1) @ Value 
    
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
                attention_f = scaled_dot_product_attention,
                decoder_mask: bool = False,
                cross_attention: bool = False
                ):

        super().__init__()
        self.d_k_q = d_k_q
        self.d_v = d_v
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.attention_f = attention_f
        self.decoder_mask = decoder_mask
        self.cross_attention = cross_attention

        self.key_mapping = nn.Linear(embedding_dim, d_k_q * heads, bias=False)
        self.query_mapping = nn.Linear(embedding_dim, d_k_q * heads, bias=False)
        self.value_mapping = nn.Linear(embedding_dim, d_v * heads, bias=False)

        self.last_linear = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)


    def forward(self, x: torch.Tensor, 
                x_attention_mask: torch.Tensor,
                x_cross: torch.Tensor = None,
                attention_cross: torch.Tensor = None,

                ):

        '''
        x -> [batch_size, sequence length, embedding_dim ]
        x_attention_mask -> [batch_size, sequence length  ]

        x_attention_mask -> mask to ignore padding tokens in attention
        decoder mask -> mask for decoder component to prevent tokens to attend subsequent ones

        x_cross -> Encoder

        '''

        Query = self.query_mapping(x)

        if self.cross_attention:
            Key = self.key_mapping(x_cross)
            Value = self.value_mapping(x_cross)
            Key = Key.reshape(x_cross.size(0), x_cross.size(1), self.heads, self.d_k_q)
            Value = Value.reshape(x_cross.size(0), x_cross.size(1), self.heads, self.d_k_q)
        else:
            Key = self.key_mapping(x)
            Value = self.value_mapping(x)
            Key = Key.reshape(x.size(0), x.size(1), self.heads, self.d_k_q)
            Value = Value.reshape(x.size(0), x.size(1), self.heads, self.d_v)

        

        Query = Query.reshape(x.size(0), x.size(1), self.heads, self.d_k_q) # Batch size, seq_len, heads, dim
        

        Key = Key.permute(0,2,1,3)  # Batch size, heads, seq_len, dim
        Query = Query.permute(0,2,1,3)
        Value = Value.permute(0,2,1,3)

        if self.decoder_mask:

            mask_decoder = torch.full((x.size(1), x.size(1)), float('-inf'))
            mask_decoder = torch.triu(mask_decoder, diagonal=1)
            '''
            mask_decoder -> [[0, -inf, -inf, -inf]
                             [0,  0,   -inf, -inf]
                             [0,  0,    0,   -inf]
                             [0,  0,    0,      0]
            
            '''
            mask_decoder = einops.repeat(mask_decoder, 'x y -> b x y', b=x.size(0)) ## triangle mask

            if self.cross_attention:

                attention = self.attention_f(Key, Query, Value, attention_cross, mask_decoder, x_attention_mask)

                print(Key.size(), Query.size())

            else:

                attention = self.attention_f(Key, Query, Value, triange_mask = mask_decoder,
                                         output_attention_mask = x_attention_mask)


        else: attention = self.attention_f(Key, Query, Value, x_attention_mask)


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

        self.attention_layers = nn.ModuleList([MultiHeadAttention(decoder_mask=True) for _ in range(stack)])

        self.cross_attention_layers = nn.ModuleList([MultiHeadAttention(decoder_mask=True, cross_attention=True) for _ in range(stack)])

        self.fedd_forward_layers = nn.ModuleList([FeedForward() for _ in range(stack)])
    

    def forward(self, x_decoder, decoder_att, x_encoder, encoder_att):

        x_decoder = self.embedding_layer(x_decoder)

        pos_embs = position_embedding(x_decoder)

        x_decoder = x_decoder + pos_embs

        print(x_decoder.size())

        for i in range(self.stack):
            x_decoder = self.attention_layers[i](x_decoder, decoder_att)
            x_decoder = self.cross_attention_layers[i](x_decoder, decoder_att, x_encoder, encoder_att)
            x_decoder = self.fedd_forward_layers[i](x_decoder)
        

        return x_decoder
        
        