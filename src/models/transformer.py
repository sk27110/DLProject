import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class ImageCaptioningModelTransformer(nn.Module):
    def __init__(self, vocab_size, vit_model, decoder_dim=256, nhead=8, num_layers=2, max_len=20, dropout=0.1):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.max_len = max_len
        self.word_embedding = nn.Embedding(vocab_size, decoder_dim)

        self.vit_model = vit_model

        self.img_proj = nn.Linear(self.vit_model.embed_dim, decoder_dim)

        decoder_layer = TransformerDecoderLayer(d_model=decoder_dim, nhead=nhead, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(decoder_dim, vocab_size)

        self.config = {
            "vocab_size": vocab_size,
            "decoder_dim": decoder_dim,
            "nhead": nhead,
            "num_layers": num_layers,
            "max_len": max_len,
            "dropout": dropout
        }

    def forward(self, images, captions, tgt_mask=None, padding_mask=None):

        img_emb = self.encoder(images)
        tgt = self.token_embedding(captions) + self.pos_encoding[:, :captions.size(1), :]

        dec_out = self.decoder(
            tgt,
            img_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask
        )

        return self.output(dec_out)
