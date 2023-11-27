import torch
from torch import nn
import math
import config



def scaled_dot_product_attention(Key, Query, Value):

    return nn.Softmax((Query @ Key.T)/math.sqrt(config.d_model)) @ Value


def position_embedding(x):

        pos_embeddings = []

        for pos, emb in enumerate(x):

            if pos % 2 == 0:
                vector = [math.sin(pos/10000**(idx/config.embedding_dim))  for idx, el in enumerate(emb)]
            else: 
                vector = [math.cos(pos/10000**(idx/config.embedding_dim))  for idx, el in enumerate(emb)]
            
            pos_embeddings.append(vector)
        
        return torch.tensor(pos_embeddings)

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
                key = nn.Parameter(torch.rand(config.d_model, config.batch_size), requires_grad=True),
                query = nn.Parameter(torch.rand(config.d_model, config.batch_size), requires_grad=True), 
                value = nn.Parameter(torch.rand(config.d_model, config.batch_size), requires_grad=True),
                d_model = config.d_model, 
                heads = config.attention_heads):

        super().__init__()

        self.key = key
        self.query = query
        self.value = value
        self.d_model = d_model
        self.heads = heads

        self.linear_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for i in range(self.heads*3)])
        self.last_linear = nn.Linear(self.d_model*self.heads, self.d_model*self.heads)
        self.layer_norm = nn.LayerNorm(self.d_model*self.heads)


    def forward(self, x):

        Key = x @ self.key.T
        Query = x @ self.query.T
        Value = x @ self.value.T

        for idx in range(0, len(self.linear_layers), 3):

            key_mapped = self.linear_layers[idx](Key)
            query_mapped = self.linear_layers[idx+2](Query)
            value_mapped = self.linear_layers[idx+1](Value)
            
            if idx == 0: concatted_attentions = scaled_dot_product_attention(key_mapped, query_mapped, value_mapped)
            else: concatted_attentions = torch.cat(concatted_attentions, scaled_dot_product_attention(key_mapped, query_mapped, value_mapped))
        

        out = self.last_linear(concatted_attentions)

        out = self.layer_norm(out + x)

        return out
    

class Encoder(nn.Module):
    def __init__(self, stack = config.encoder_stack):
        super().__init__()

        self.stack = stack

        self.attention_model = MultiHeadAttention()
        self.ff_model = FeedForward()

    def forward(self, x):

        pos_embs = position_embedding(x)

        x = x + pos_embs

        for _ in range(self.stack):
            x = self.attention_model(x)
            x = self.ff_model(x)
    
        return x

        
    
    
            

                

        


        
        



        

        

           
        


    





