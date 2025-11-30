import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import timm


class ImageCaptioningModelTransformer(nn.Module):
    def __init__(self, vocab_size, decoder_dim=256, nhead=8, num_layers=2, max_len=20, dropout=0.1):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.max_len = max_len
        self.word_embedding = nn.Embedding(vocab_size, decoder_dim)


        vit_model = timm.create_model('vit_small_patch16_224', pretrained=True)
        vit_model.head = torch.nn.Identity()
        for p in vit_model.parameters():
            p.requires_grad = False

        vit_model.eval()

        self.vit_model = vit_model

        self.img_proj = nn.Linear(self.vit_model.embed_dim, decoder_dim)

        decoder_layer = TransformerDecoderLayer(d_model=decoder_dim, nhead=nhead, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(decoder_dim, vocab_size)

    def forward(self, images, captions, tgt_mask=None):
        batch_size = images.size(0)
        img_features = self.vit_model(images)         
        img_emb = self.img_proj(img_features) 
        img_emb = img_emb.unsqueeze(0)
        tgt_emb = self.word_embedding(captions).permute(1, 0, 2)

        out = self.transformer_decoder(tgt=tgt_emb, memory=img_emb, tgt_mask=tgt_mask)

        out = self.fc_out(out)

        return out.permute(1, 0, 2)