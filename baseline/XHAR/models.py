"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""
from scipy.stats import special_ortho_group
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

from utils import split_last, merge_last, _rotate_random

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."

    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        # return x


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg, pos_embed=None):
        super().__init__()
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)

        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden)  # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)  # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        # return self.drop(self.norm(e))
        return self.norm(e)


class Projecter(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        self.norm = LayerNorm(cfg)

    def forward(self, x):
        # factorized embedding
        e = self.lin(x)
        return self.norm(e)


class EmbeddingsA(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden)  # position embedding
        else:
            self.pos_embed = pos_embed
        self.emb_norm = cfg.emb_norm
        self.norm = LayerNorm(cfg)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)  # (S,) -> (B, S)

        e = x + self.pos_embed(pos)
        # return self.drop(self.norm(e))
        return self.norm(e)

class MultiProjection(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        return q, k, v


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None  # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        # scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, cfg, embed=None):
        super().__init__()
        if embed is None:
            self.embed = Embeddings(cfg)
        else:
            self.embed = embed

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)

    def forward(self, x):
        h = self.embed(x)

        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class BaseModule(nn.Module):

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)


class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm = LayerNorm(cfg)
        self.pred = nn.Linear(cfg.hidden, cfg.feature_num)

    def forward(self, input_seqs):
        h_masked = gelu(self.linear(input_seqs))
        h_masked = self.norm(h_masked)
        return self.pred(h_masked)


class BertModel4Pretrain(nn.Module):

    def __init__(self, cfg, output_embed=False, freeze_encoder=False, freeze_decoder=False):
        super().__init__()
        self.encoder = Transformer(cfg)
        if freeze_encoder:
            freeze(self.encoder)
        self.decoder = Decoder(cfg)
        if freeze_decoder:
            freeze(self.decoder)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.encoder(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        return self.decoder(h_masked)


class BertAdaModel4Pretrain(BaseModule):

    def __init__(self, cfg, output_embed=False, adapter_range=[0.9, 1.1]):
        super().__init__()
        self.encoder = Transformer(cfg)
        self.decoder = Decoder(cfg)
        self.output_embed = output_embed
        self.adapter = nn.Parameter(torch.ones(cfg.feature_num), requires_grad=True)
        self.adapter_bias = nn.Parameter(torch.zeros(cfg.feature_num), requires_grad=True)
        self.adapter_range = adapter_range

    def forward(self, input_seqs, masked_pos=None, adapt=False):
        if adapt:
            input_seqs = self.adapt_feature_weights(input_seqs)
        h_masked = self.encoder(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        return self.decoder(h_masked)

    def adapt_feature_weights(self, input_seqs):
        fw = torch.clip(self.adapter, self.adapter_range[0], self.adapter_range[1])
        return (input_seqs - self.adapter_bias) * fw

    def freeze_encoder_decoder(self):
        freeze(self.encoder)
        freeze(self.decoder)

    def freeze_adapter(self):
        self.adapter.requires_grad = False
        self.adapter_bias.requires_grad = False

    def unfreeze_all(self):
        unfreeze(self.encoder)
        unfreeze(self.decoder)
        self.adapter.requires_grad = True
        self.adapter_bias.requires_grad = True



class BertPerModel4Pretrain(BaseModule):

    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.adapter = Projecter(cfg)
        self.encoder = Transformer(cfg, embed=EmbeddingsA(cfg))
        self.decoder = Decoder(cfg)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h = self.adapter(input_seqs)
        h_masked = self.encoder(h)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        return self.decoder(h_masked)


class CompositeClassifier(BaseModule):

    def __init__(self, encoder_cfg, classifier=None, freeze_encoder=False, freeze_classifier=False, output_embed=False):
        super().__init__()
        self.encoder = Transformer(encoder_cfg)
        if freeze_encoder:
            freeze(self.encoder)
        self.classifier = classifier
        if freeze_classifier:
            freeze(self.classifier)
        self.output_embed = output_embed

    def forward(self, input_seqs, training=False):
        h = self.encoder(input_seqs)
        # if self.output_embed:
        #     return h
        h = self.classifier(h, training, self.output_embed)
        return h


class CompositePerClassifier(BaseModule):

    def __init__(self, encoder_cfg, adapter_num, adapters, classifier=None,
                 freeze_encoder=False, freeze_classifier=False, output_embed=False):
        super().__init__()
        self.adapter = Projecter(encoder_cfg)
        freeze(self.adapter)
        self.adapters = adapters
        for i in range(adapter_num):
            adapter_i = Projecter(encoder_cfg)
            freeze(adapter_i)
            self.__setattr__('adapter' + str(i), adapter_i)
        self.encoder = Transformer(encoder_cfg, embed=EmbeddingsA(encoder_cfg))
        self.decoder = Decoder(encoder_cfg)
        self.output_embed = output_embed
        if freeze_encoder:
            freeze(self.encoder)
        self.classifier = classifier
        if freeze_classifier:
            freeze(self.classifier)
        self.output_embed = output_embed

    def forward(self, input_seqs, training=False, user_label=None):
        if user_label is None:
            h = self.adapter(input_seqs)
        else:
            h = []
            for i in range(input_seqs.size(0)):
                adapter = self.__getattr__('adapter' + str(int(user_label[i])))
                h.append(adapter(input_seqs[i]))
            h = torch.stack(h, dim=0)
            # h.requires_grad = True
            # h = Variable(h, requires_grad=True)
        h = self.encoder(h)
        if self.output_embed:
            return h
        h = self.classifier(h, training, self.output_embed)
        return h

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)
        self.load_state_dict(self.adapters, strict=False)


class DAV1(BaseModule):

    def __init__(self, ae_cfg, classifier=None, output_embed=False, freeze_encoder=False, freeze_decoder=False
                 , freeze_classifier=False):
        super().__init__()
        self.encoder = Transformer(ae_cfg)
        if freeze_encoder:
            freeze(self.encoder)
        self.decoder = Decoder(ae_cfg)
        if freeze_decoder:
            freeze(self.decoder)
        self.classifier = classifier
        if freeze_classifier:
            freeze(self.classifier)
        self.output_embed = output_embed

    def forward(self, input_seqs, training=False, output_clf=True, masked_pos=None, embed=False):
        h = self.encoder(input_seqs)
        if self.output_embed or embed:
            return h
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
            h = torch.gather(h, 1, masked_pos)
        if output_clf:
            c = self.classifier(h, training)
            return c
        else:
            r = self.decoder(h)
            return r


class DAV2(BaseModule):

    def __init__(self, ae_cfg, classifier=None, output_embed=False, freeze_adapter=True
                 , freeze_decoder=False, freeze_classifier=False):
        super().__init__()
        self.encoder = Transformer(ae_cfg)
        self.decoder = Decoder(ae_cfg)
        if freeze_decoder:
            freeze(self.decoder)
        self.classifier = classifier
        if freeze_classifier:
            freeze(self.classifier)
        self.adapter = nn.Parameter(torch.ones(ae_cfg.feature_num), requires_grad=not freeze_adapter)
        self.output_embed = output_embed

    def forward(self, input_seqs, training=False, output_clf=True, masked_pos=None, embed=False, adapt=False):
        h = input_seqs
        if adapt:
            h = self.adapt_feature_weights(h)
        h = self.encoder(h)
        if self.output_embed or embed:
            return h
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
            h = torch.gather(h, 1, masked_pos)
        if output_clf:
            c = self.classifier(h, training)
            return c
        else:
            r = self.decoder(h)
            return r

    def adapt_feature_weights(self, input_seqs):
        fw = torch.clip(self.adapter, 0.9, 1.1)
        return input_seqs * fw


class DAV3(BaseModule):

    def __init__(self, ae_cfg, classifier=None, output_embed=False, freeze_adapter=True
                 , freeze_decoder=False, freeze_classifier=False, seq_len=None):
        super().__init__()
        self.encoder = Transformer(ae_cfg)
        self.decoder = Decoder(ae_cfg)
        self.classifier = classifier
        if freeze_decoder:
            freeze(self.decoder)
        self.classifier = classifier
        if freeze_classifier:
            freeze(self.classifier)
        self.feature_weights = nn.Parameter(torch.ones(ae_cfg.feature_num), requires_grad=not freeze_adapter)
        if seq_len is None: seq_len = ae_cfg.seq_len
        self.temporal_weights = nn.Parameter(torch.ones(seq_len), requires_grad=not freeze_adapter)
        self.output_embed = output_embed

    def forward(self, input_seqs, training=False, output_clf=True, masked_pos=None, embed=False, adapt=False):
        h = input_seqs
        if adapt:
            h = self.adapt_feature_weights(h)
            h = self.adapt_temporal_weights(h)
        h = self.encoder(h)
        if self.output_embed or embed:
            return h
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
            h = torch.gather(h, 1, masked_pos)
        if output_clf:
            c = self.classifier(h, training)
            return c
        else:
            r = self.decoder(h)
            return r

    def adapt_feature_weights(self, input_seqs):
        fw = torch.clip(self.feature_weights, 0.9, 1.1)
        return input_seqs * fw

    def adapt_temporal_weights(self, input_seqs):
        seqs = torch.transpose(input_seqs, 1, 2)
        fw = torch.clip(self.temporal_weights, 0.9, 1.1)
        return torch.transpose(seqs * fw, 1, 2)


class Adapter(nn.Module):
    def __init__(self, feature_num, seq_len):
        super().__init__()
        self.adapter_w1 = nn.Parameter(torch.ones(feature_num, feature_num))
        self.adapter_b1 = nn.Parameter(torch.ones(feature_num))


class DAV4(BaseModule):

    def __init__(self, ae_cfg, classifier_cfg, classifier=None, encoder_source=None, output_embed=False
                 , freeze_adapter=True, freeze_target_model=True):
        super().__init__()
        self.encoder_source = Transformer(ae_cfg)
        if encoder_source is not None:
            self.encoder_source = encoder_source
        self.encoder = Transformer(ae_cfg)
        self.decoder = Decoder(ae_cfg)
        if freeze_target_model:
            self.freeze_target_model()
        self.classifier = classifier
        self.feature_weights = nn.Parameter(torch.ones(ae_cfg.feature_num), requires_grad=not freeze_adapter)
        self.feature_adapter_1 = nn.Linear(ae_cfg.feature_num, ae_cfg.feature_num)
        self.feature_adapter_2 = nn.Linear(classifier_cfg.seq_len, classifier_cfg.seq_len)
        self.embed_adapter = nn.Linear(ae_cfg.hidden, ae_cfg.hidden)
        if freeze_adapter:
            self.freeze_adapters()
        self.output_embed = output_embed

    def forward(self, input_seqs, training=False, output_clf=True, masked_pos=None, embed=False
                , adapt_feature=False, adapt_weight=False, adapt_embed=False, source_encoder=True):
        if adapt_feature:
            input_seqs = self.adapt_feature(input_seqs)
        if adapt_weight:
            input_seqs = self.adapt_feature_weights(input_seqs)
        if source_encoder:
            h = self.encoder_source(input_seqs)
        else:
            h = self.encoder(input_seqs)
        if adapt_embed:
            h = self.embed_adapter(h)
        if self.output_embed or embed:
            return h
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
            h = torch.gather(h, 1, masked_pos)
        if output_clf:
            c = self.classifier(h, training)
            return c
        else:
            r = self.decoder(h)
            return r

    def adapt_feature(self, features):
        af = self.feature_adapter_1(features)
        af = torch.transpose(af, 1, 2)
        af = self.feature_adapter_2(af)
        af = torch.transpose(af, 1, 2)
        af = torch.max(torch.min(af, features * 1.5), features * 0.5)
        return af

    def adapt_feature_weights(self, input_seqs):
        fw = torch.clamp(self.feature_weights, 0.9, 1.1)
        return input_seqs * fw

    def freeze_target_model(self):
        freeze(self.encoder)
        freeze(self.decoder)

    def freeze_adapters(self):
        freeze(self.feature_adapter_1)
        freeze(self.feature_adapter_2)
        freeze(self.embed_adapter)




class ClassifierGRU(BaseModule):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i]
                                        , bidirectional=cfg.rnn_bidirection[i], batch_first=True))
            else:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i]
                                        , bidirectional=cfg.rnn_bidirection[i], batch_first=True))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear
        self.bidirectional = any(cfg.rnn_bidirection)

    def forward(self, input_seqs, training=False, embed=False):
        h = input_seqs
        for i in range(self.num_rnn):
            rnn = self.__getattr__('gru' + str(i))
            h, _ = rnn(h)
            if self.activ:
                h = F.relu(h)
        # if self.bidirectional:
        #     h = h.view(h.size(0), h.size(1), 2, -1)
        #     h = torch.cat([h[:, -1, 0], h[:, 0, 1]], dim=1)
        # else:
        h = h[:, -1, :]
        if embed:
            return h
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class ClassifierAdaGRU(BaseModule):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i]
                                        , bidirectional=cfg.rnn_bidirection[i], batch_first=True))
            else:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i]
                                        , bidirectional=cfg.rnn_bidirection[i], batch_first=True))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear
        self.bidirectional = any(cfg.rnn_bidirection)

    def forward(self, input_seqs, training=False, embed=False):
        h = input_seqs
        for i in range(self.num_rnn):
            rnn = self.__getattr__('gru' + str(i))
            h, _ = rnn(h)
            if self.activ:
                h = F.relu(h)
        if self.bidirectional:
            h = torch.cat([h[:, -1, :], h[:, 0, :]], dim=1)
        else:
            h = h[:, -1, :]
        if embed:
            return h
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


class BenchmarkDCNN(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, (5, 1))
        self.bn1 = nn.BatchNorm2d(50)
        self.conv2 = nn.Conv2d(50, 40, (5, 1))
        self.bn2 = nn.BatchNorm2d(40)
        if cfg.seq_len <= 20:
            self.conv3 = nn.Conv2d(40, 20, (2, 1))
        else:
            self.conv3 = nn.Conv2d(40, 20, (3, 1))
        self.bn3 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d((2, 1))
        self.lin1 = nn.Linear(input * cfg.flat_num, 400)
        self.lin2 = nn.Linear(400, output)

    def forward(self, input_seqs, training=False):
        h = input_seqs.unsqueeze(1)
        h = F.relu(torch.tanh(self.conv1(h)))
        h = self.bn1(self.pool(h))
        h = F.relu(torch.tanh(self.conv2(h)))
        h = self.bn2(self.pool(h))
        h = F.relu(torch.tanh(self.conv3(h)))
        h = h.view(h.size(0), h.size(1), h.size(2) * h.size(3))
        h = self.lin1(h)
        h = F.relu(torch.tanh(torch.sum(h, dim=1)))
        h = self.normalize(h[:, :, None, None])
        h = self.lin2(h[:, :, 0, 0])
        return h

    def normalize(self, x, k=1, alpha=2e-4, beta=0.75):
        # x = x.view(x.size(0), x.size(1) // 5, 5, x.size(2), x.size(3))#
        # y = x.clone()
        # for s in range(x.size(0)):
        #     for j in range(x.size(1)):
        #         for i in range(5):
        #             norm = alpha * torch.sum(torch.square(y[s, j, i, :, :])) + k
        #             norm = torch.pow(norm, -beta)
        #             x[s, j, i, :, :] = y[s, j, i, :, :] * norm
        # x = x.view(x.size(0), x.size(1) * 5, x.size(3), x.size(4))
        return x


class BenchmarkDeepSense(nn.Module):

    def __init__(self, cfg, input=None, output=None, num_filter=8):
        super().__init__()
        self.sensor_num = input // 3
        for i in range(self.sensor_num):
            self.__setattr__('conv' + str(i) + "_1", nn.Conv2d(1, num_filter, (2, 3)))
            self.__setattr__('conv' + str(i) + "_2", nn.Conv2d(num_filter, num_filter, (3, 1)))
            self.__setattr__('conv' + str(i) + "_3", nn.Conv2d(num_filter, num_filter, (2, 1)))
            self.__setattr__('bn' + str(i) + "_1", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_2", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_3", nn.BatchNorm2d(num_filter))
        self.conv1 = nn.Conv2d(1, num_filter, (2, self.sensor_num))
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, (3, 1))
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.conv3 = nn.Conv2d(num_filter, num_filter, (2, 1))
        self.bn3 = nn.BatchNorm2d(num_filter)
        self.flatten = nn.Flatten()

        self.lin1 = nn.Linear(cfg.flat_num, 12)
        self.lin2 = nn.Linear(12, output)

    def forward(self, input_seqs, training=False):
        h = input_seqs.view(input_seqs.size(0), input_seqs.size(1), self.sensor_num, 3)
        hs = []
        for i in range(self.sensor_num):
            t = h[:, :, i, :]
            t = torch.unsqueeze(t, 1)
            for j in range(3):
                cv = self.__getattr__('conv' + str(i) + "_" + str(j + 1))
                bn = self.__getattr__('bn' + str(i) + "_" + str(j + 1))
                t = bn(F.relu(cv(t)))
            hs.append(self.flatten(t)[:, :, None])
        h = torch.cat(hs, dim=2)
        h = h.unsqueeze(1)
        h = self.bn1(F.relu(self.conv1(h)))
        h = self.bn2(F.relu(self.conv2(h)))
        h = self.bn3(F.relu(self.conv3(h)))
        h = self.flatten(h)
        h = self.lin2(F.relu(self.lin1(h)))
        return h


class BenchmarkHDCNN(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.conv1 = nn.Conv1d(input, 32, 5)
        self.conv2 = nn.Conv1d(32, 64, 5)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(cfg.flat_num, 128)
        self.lin2 = nn.Linear(128, output)

    def forward(self, input_seqs, training=False):
        h = input_seqs.transpose(1, 2)
        h1 = F.relu(self.pool(self.conv1(h)))
        h2 = F.relu(self.pool(self.conv2(h1)))
        h3 = self.flatten(h2)
        h3 = F.relu(self.lin1(h3))
        h = F.relu(self.lin2(h3))
        if training:
            return h, [h1, h2, h3]
        else:
            return h


### XHAR ###
# Following codes are from the source codes of XHAR, thanks for the share from the authors.
##

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class BenchmarkXHAR(nn.Module):

    def __init__(self, cfg, input=None, output=None):
        super().__init__()

        self.feature = nn.Sequential()  # 32,51,4
        self.feature.add_module('f_conv1', nn.Conv1d(input, 64, kernel_size=2))  # batch_size, dim, time-1 #64,3
        self.feature.add_module('f_bn1', nn.BatchNorm1d(64))  # batch_size, dim, time-1
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))  # batch_size, dim/2, (time-1)/2
        self.feature.add_module('f_relu1', nn.ReLU(True))

        self.lstm = nn.GRU(input - 1, 100, num_layers=3, batch_first=True,
                           bidirectional=True)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(464, 200))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(200))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(200, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, output))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(464, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))

        self.cnn_domain_classifier = nn.Sequential()
        self.cnn_domain_classifier.add_module('cnn_d_fc1', nn.Linear(64, 10))
        self.cnn_domain_classifier.add_module('cnn_d_bn1', nn.BatchNorm1d(10))
        self.cnn_domain_classifier.add_module('cnn_d_relu1', nn.ReLU(True))
        self.cnn_domain_classifier.add_module('cnn_d_fc2', nn.Linear(10, 2))

        self.gru_domain_classifier = nn.Sequential()
        self.gru_domain_classifier.add_module('gru_d_fc1', nn.Linear(400, 100))
        self.gru_domain_classifier.add_module('gru_d_bn1', nn.BatchNorm1d(100))
        self.gru_domain_classifier.add_module('gru_d_relu1', nn.ReLU(True))
        self.gru_domain_classifier.add_module('gru_d_fc2', nn.Linear(100, 2))


    '''
     input_data: [batch_size, time, dim]  [32, 4, 51]
    '''

    def forward(self, input_data, training=False, lam=1.0):
        feature_gru, _ = self.lstm(input_data[:, :, :-1])  # 32,4,100

        f_head = feature_gru[:, 0, :]  # 32,1,100
        f_tail = feature_gru[:, feature_gru.shape[1] - 1, :]  # 32,1,100
        feature_gru_cat = torch.cat((f_head, f_tail), dim=-1)  # 32,200

        feature_cnn = self.feature(torch.transpose(input_data, 1, 2))  # batch_size, 32, 1
        feature_cnn = feature_cnn.reshape((feature_cnn.shape[0], -1))  # batch_size, 32
        feature_cat = torch.cat((feature_gru_cat, feature_cnn), dim=-1)  # # batch_size, 32+2*100

        class_output = self.class_classifier(feature_cat)

        if training is True:
            reverse_feature_gru_cat = GradientReversalFunction.apply(feature_gru_cat, lam)
            gru_domain_classifier = self.gru_domain_classifier(reverse_feature_gru_cat)

            reverse_feature_cnn = GradientReversalFunction.apply(feature_cnn, lam)
            cnn_domain_classifier = self.cnn_domain_classifier(reverse_feature_cnn)

            # attention, feature_attention = self.self_attention(feature_gru) #[batch_size, 2*lstm_hid_dim]

            reverse_feature = GradientReversalFunction.apply(feature_cat, lam)
            domain_output = self.domain_classifier(reverse_feature)
            return class_output, domain_output, gru_domain_classifier, cnn_domain_classifier
        else:
            return class_output



def fetch_classifier(method, model_cfg, input=None, output=None):
    if 'agru' in method:
        model = ClassifierAdaGRU(model_cfg, input=input, output=output)
    elif 'gru' in method:
        model = ClassifierGRU(model_cfg, input=input, output=output)
    elif 'hdcnn' in method:
        model = BenchmarkHDCNN(model_cfg, input=input, output=output)
    elif 'dcnn' in method:
        model = BenchmarkDCNN(model_cfg, input=input, output=output)
    elif 'deepsense' in method:
        model = BenchmarkDeepSense(model_cfg, input=input, output=output)
    elif 'xhar' in method:
        model = BenchmarkXHAR(model_cfg, input=input, output=output)
    else:
        model = None
    return model
