import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

class ImageEncoder(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, out_dim=512, train_backbone=False):
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(net.children())[:-1])
            in_dim = 2048
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            self.backbone = net.features
            in_dim = 1280
        else:
            raise ValueError("backbone must be resnet50 or efficientnet_b0")
        for p in self.backbone.parameters():
            p.requires_grad = train_backbone
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    def forward(self, x):
        feat = self.backbone(x)
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        else:
            feat = feat.flatten(1)
        return self.proj(feat)

class BiLSTMTextEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=256, bidirectional=True, pad_idx=0, out_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_out = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(lstm_out, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    def forward(self, ids, lengths):
        x = self.embed(ids)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        return self.proj(h)

class ConcatFusion(nn.Module):
    def __init__(self, dim, out_dim=512):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim * 2, out_dim), nn.ReLU(), nn.Dropout(0.2))
    def forward(self, img_feat, txt_feat):
        return self.fc(torch.cat([img_feat, txt_feat], dim=1))

class ElementWiseFusion(nn.Module):
    def __init__(self, dim, out_dim=512):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, out_dim), nn.ReLU(), nn.Dropout(0.2))
    def forward(self, img_feat, txt_feat):
        return self.fc(img_feat * txt_feat)

class CoAttentionFusion(nn.Module):
    def __init__(self, dim, out_dim=512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(dim, out_dim), nn.ReLU(), nn.Dropout(0.2))
    def forward(self, img_feat, txt_feat):
        g = self.gate(torch.cat([img_feat, txt_feat], dim=1))
        fused = g * img_feat + (1 - g) * txt_feat
        return self.fc(fused)

class LSTMAnswerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.init_h = nn.Linear(hidden_dim, hidden_dim)
        self.init_c = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, fused, answer_inp):
        x = self.embed(answer_inp)
        h0 = torch.tanh(self.init_h(fused)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(fused)).unsqueeze(0)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc_out(out)
    @torch.no_grad()
    def greedy_decode(self, fused, bos_idx, eos_idx, max_len=12):
        B = fused.size(0)
        h = torch.tanh(self.init_h(fused)).unsqueeze(0)
        c = torch.tanh(self.init_c(fused)).unsqueeze(0)
        cur = torch.full((B, 1), bos_idx, dtype=torch.long, device=fused.device)
        outs = []
        for _ in range(max_len):
            x = self.embed(cur[:, -1:])
            out, (h, c) = self.lstm(x, (h, c))
            logit = self.fc_out(out[:, -1])
            nxt = logit.argmax(dim=-1, keepdim=True)
            outs.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
        return torch.cat(outs, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerAnswerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2, pad_idx=0, max_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.memory_proj = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, fused, answer_inp):
        tgt = self.pos(self.embed(answer_inp))
        memory = self.memory_proj(fused).unsqueeze(1)
        tgt_mask = torch.triu(torch.ones(answer_inp.size(1), answer_inp.size(1), device=answer_inp.device), diagonal=1).bool()
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(out)
    @torch.no_grad()
    def greedy_decode(self, fused, bos_idx, eos_idx, max_len=12):
        B = fused.size(0)
        out_tokens = torch.full((B, 1), bos_idx, dtype=torch.long, device=fused.device)
        for _ in range(max_len):
            logits = self.forward(fused, out_tokens)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            out_tokens = torch.cat([out_tokens, nxt], dim=1)
            if (nxt.squeeze(1) == eos_idx).all():
                break
        return out_tokens[:, 1:]

class VQAModelA(nn.Module):
    def __init__(
        self,
        q_vocab_size,
        a_vocab_size,
        image_backbone="resnet50",
        fusion_type="coattention",
        decoder_type="lstm",
        feat_dim=512,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(backbone=image_backbone, out_dim=feat_dim)
        self.text_encoder = BiLSTMTextEncoder(q_vocab_size, out_dim=feat_dim, pad_idx=0)
        if fusion_type == "concat":
            self.fusion = ConcatFusion(feat_dim, out_dim=feat_dim)
        elif fusion_type == "elementwise":
            self.fusion = ElementWiseFusion(feat_dim, out_dim=feat_dim)
        elif fusion_type == "coattention":
            self.fusion = CoAttentionFusion(feat_dim, out_dim=feat_dim)
        else:
            raise ValueError("fusion_type phải là concat / elementwise / coattention")
        if decoder_type == "lstm":
            self.decoder = LSTMAnswerDecoder(a_vocab_size, hidden_dim=feat_dim, pad_idx=0)
        elif decoder_type == "transformer":
            self.decoder = TransformerAnswerDecoder(a_vocab_size, d_model=feat_dim, pad_idx=0)
        else:
            raise ValueError("decoder_type phải là lstm / transformer")
        self.decoder_type = decoder_type
    def forward(self, images, q_ids, q_lens, a_inp):
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(q_ids, q_lens)
        fused = self.fusion(img_feat, txt_feat)
        return self.decoder(fused, a_inp)
    @torch.no_grad()
    def generate(self, images, q_ids, q_lens, bos_idx, eos_idx, max_len=12):
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(q_ids, q_lens)
        fused = self.fusion(img_feat, txt_feat)
        return self.decoder.greedy_decode(fused, bos_idx, eos_idx, max_len=max_len)
