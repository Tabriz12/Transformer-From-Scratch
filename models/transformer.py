import torch
from torch import nn
from models.encoder import Encoder, Decoder


class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    

    def forward(self, en: torch.Tensor, 
                en_attention: torch.Tensor, 
                tr: torch.Tensor, 
                tr_attention: torch.Tensor):

        encoder_out = self.encoder(en, en_attention)

        return encoder_out
