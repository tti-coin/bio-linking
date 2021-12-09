import collections
import contextlib
from typing import Optional, Dict
import tqdm
import dataclasses

import numpy as np
import torch
import torch.nn.functional as F
import transformers


def max_pool(inputs, masks):
    # inputs.shape = [batch_size, seq_len, dim]
    # masks.shape = [batch_size, seq_len]
    shift = -inputs.min() + 1.0
    return ((inputs + shift) * masks[:,:,None]).max(1)[0] - shift

def average_pool(inputs, masks, eps=1e-10):
    return (inputs * masks[:,:,None]).sum(1) / (masks.sum(1)[:,None]+eps)


class BertEncoder(torch.nn.Module):
    def __init__(self, bert_name_or_path):
        super(BertEncoder, self).__init__()
        self.bert = transformers.AutoModel.from_pretrained(bert_name_or_path)
        self.output_dim = self.bert.config.hidden_size
    def forward(self, inputs, input_mask, pool_mask):
        h = self.bert(inputs, attention_mask=input_mask).last_hidden_state
        #h = max_pool(h, pool_mask)
        h = average_pool(h, pool_mask)
        return h # [batch_size, output_dim]

class LSTMEncoder(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, pad_id, cell_size):
        raise NotImplementedError("LSTMEncoder")
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.pad_id = pad_id
        self.emb_word = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=self.pad_id)
        self.output_dim = 2*cell_size
        self.rnn = torch.nn.LSTM(self.emb_dim, self.cell_size, bidirectional=True)

    def forward(self, inputs, input_mask, pool_mask):
        seq_lens = input_mask.sum(1).to(torch.long)
        h = self.emb_word(inputs)

        packed = torch.nn.utils.rnn.pack_padded_sequence(h, seq_lens, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True) # [batch_size, seq_len, hidden_dim]

        #h = max_pool(h, pool_mask)
        h = average_pool(h, pool_mask)
        return h # [batch_size, output_dim]


class LinearOutput(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearOutput, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim-1)
        self.no_concept_margin = torch.nn.Parameter(torch.FloatTensor(1).zero_(), requires_grad=False)
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        logits = self.linear(inputs)
        logits = torch.cat([self.no_concept_margin[None].repeat(batch_size,1), logits], 1)
        return logits

    # TODO: debug
    #@property
    #def weight(self):
    #    return self.linear.weight

class CosineOutput(torch.nn.Module):
    def __init__(self, input_dim, output_dim, scale):
        super(CosineOutput, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = torch.nn.Parameter(torch.FloatTensor(output_dim-1, input_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.no_concept_margin = torch.nn.Parameter(torch.FloatTensor(1).zero_())
        self.scale = scale
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        cosine = torch.cat([self.no_concept_margin[None].repeat(batch_size,1), cosine], 1)
        cosine = cosine * self.scale
        return cosine

    def overwrite_weight(self, new_weights:Dict[int,np.ndarray]) -> None:
        dtype = self.weight.dtype
        for key in new_weights.keys():
            self.weight.data[key-1] = torch.tensor(new_weights[key], dtype=dtype)

class ArcFaceMargin(torch.nn.Module):
    def __init__(self, margin, scale):
        super(ArcFaceMargin, self).__init__()
        self.margin = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        torch.nn.init.constant_(self.margin, margin)
        self.cos_margin = torch.nn.Parameter(torch.cos(self.margin).clone().detach(), requires_grad=False)
        self.sin_margin = torch.nn.Parameter(torch.sin(self.margin).clone().detach(), requires_grad=False)
        self.scale = scale

    def forward(self, scaled_cosine, golds):
        # cosine: [batch_size, num_class]
        # golds: [batch_size]
        # margin: scalar
        cosine = scaled_cosine / self.scale
        positive_coss = cosine.gather(1, golds[:,None]) # [batch_size, 1]
        positive_sins = (1-positive_coss.pow(2)).clamp(min=0.0).sqrt()

        margined = positive_coss * self.cos_margin - positive_sins * self.sin_margin # [batch_size, 1]
        diff = (margined - positive_coss) # [batch_size, 1]
        diff = torch.zeros_like(cosine).scatter(1, golds[:,None], diff) # [batch_size, num_class]

        output = (cosine + diff.detach()) * self.scale
        return output



@dataclasses.dataclass
class ModelConfig:
    encoder: str # ["bert", "lstm"]
    encoder_pool_target: str # ["entity", "input"]
    feature_dim: int
    num_class: int
    dropout_output_rate: float
    use_feature_bias: bool = False
    use_feature_following_linear: bool = False

    cosine: bool = True
    cosine_scale: Optional[int] = None
    arcface_margin: Optional[float] = None

    # bert encoder
    bert_name_or_path: Optional[str] = None

    # lstm encoder
    vocab_size: Optional[int] = None
    emb_dim: Optional[int] = None
    pad_id: Optional[int] = None
    cell_size: int = None

class Model(torch.nn.Module):
    def __init__(self, config:ModelConfig):
        super(Model, self).__init__()
        self.config = config

        assert config.encoder in ["bert", "lstm"]
        if config.encoder == "bert":
            self.encoder = BertEncoder(bert_name_or_path=config.bert_name_or_path)
        elif config.encoder == "lstm":
            self.encoder = LSTMEncoder(vocab_size=config.vocab_size, emb_dim=config.emb_dim, pad_id=config.pad_id, cell_size=config.cell_size)

        self.fc_feature = torch.nn.Linear(self.encoder.output_dim, config.feature_dim, bias=config.use_feature_bias)
        if config.use_feature_following_linear:
            self.fc_feature_follow = torch.nn.Linear(config.feature_dim, config.feature_dim)
        if config.dropout_output_rate > 0.0:
            self.dropout_output = torch.nn.Dropout(config.dropout_output_rate)
        else:
            self.dropout_output = torch.nn.Identity()

        if config.cosine:
            self.output = CosineOutput(config.feature_dim, config.num_class, scale=config.cosine_scale)
            self.arcface = ArcFaceMargin(margin=config.arcface_margin, scale=config.cosine_scale)
        else:
            self.output = LinearOutput(config.feature_dim, config.num_class)

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    def extract_features(self, inputs, input_mask, entity_mask=None):
        if self.config.encoder_pool_target == "entity":
            pool_mask = entity_mask
        elif self.config.encoder_pool_target == "input":
            pool_mask = input_mask
        h = self.encoder(inputs=inputs, input_mask=input_mask, pool_mask=pool_mask)
        h = self.fc_feature(h).tanh()
        if self.config.use_feature_following_linear:
            h = self.fc_feature_follow(h)

        if self.config.cosine:
            h = F.normalize(h)

        return h # [B, feature_dim]

    def forward(self, inputs, input_mask, entity_mask=None, golds=None):
        outputs = dict()
        h = self.extract_features(inputs=inputs, input_mask=input_mask, entity_mask=entity_mask)

        h = self.dropout_output(h)
        logits = self.output(h)
        outputs["logits"] = logits
        if self.config.cosine and self.training:
            arcface_logits = self.arcface(logits, golds)
            outputs["arcface_logits"] = arcface_logits
        return outputs


