import torch
from torch import nn
from models.encoder_decoder import Encoder, Decoder
import config
from torch.nn import functional as F


class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_linear = nn.Linear(config.embedding_dim, config.voc_size)
    

    def forward(self, en: torch.Tensor, 
                en_attention: torch.Tensor, 
                tr: torch.Tensor, 
                tr_attention: torch.Tensor):

        encoder_out = self.encoder(en, en_attention)

        decoder_out = self.decoder(tr, tr_attention, encoder_out, en_attention)

        out = self.final_linear(decoder_out)

        return out
