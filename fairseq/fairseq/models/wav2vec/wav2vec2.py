# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
    FairseqDropout
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange, index_put, is_xla_tensor
from fairseq.modules.adapter import LightAdapter
from fairseq.data.data_utils import lengths_to_padding_mask

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass
class Wav2Vec2Config(FairseqDataclass):
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
                    "groups in the first conv block, whereas layer_norm has layer norms in "
                    "every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )

    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
                    "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
                    "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    quantize_targets: bool = field(
        default=False, metadata={"help": "use quantized targets"}
    )
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )
    same_quantizer: bool = field(
        default=False, metadata={"help": "use same quantizer for inputs and targets"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
                    "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
                    "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
                    "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
                    "can be tuple of 3 values (start, end, decay)"
        },
    )
    pool_positions: str = field(
        default="3,6,9",
        metadata={
            "help": "string describing the positions of pooling layers in form of a python list that contains "
                    "[int1, int2, int3]"
        },
    )

    pool_mode: str = field(
        default="layer_norm",
        metadata={
            "help": "pooling mode."
        },
    )
    pool_op: str = field(
        default="conv",
        metadata={
            "help": "pooling operation; conv, min, or max"
        },
    )
    pool_norm: bool = field(
        default=True,
        metadata={"help": "whether use layer_norm in pool layer; for ablation study"},
    )

    pool_layer: str = field(
        default="1024,3,2",
        metadata={
            "help": "string describing cnn pooling layers in form of a python list that contains "
                    "[(dim, kernel_size, stride), ...]"
        },
    )

    reset_top_layers: int = field(
        default=-1, metadata={"help": "length of the mask for features (channels)"}
    )

    pos_positions: str = field(
        default="",
        metadata={
            "help": "string describing the positions of pos layers in form of a python list that contains "
                    "[int1, int2, int3]"
        },
    )
    share_pos_pool_pos: bool = field(
        default=False,
        metadata={"help": "the same positions for pos and pool"},
    )

    extra_pos: bool = field(
        default=False,
        metadata={"help": "use positional encoding together with pooling layer"},
    )
    extra_pos_before_pool: bool = field(
        default=False,
        metadata={"help": "use positional encoding right before each injected pooling layer"},
    )

    pos_kaiming_init: bool = field(
        default=False,
        metadata={"help": "whether to use kaiming init to initialize position layer"},
    )
    pos_gelu: bool = field(
        default=True,
        metadata={"help": "whether to use gelu in position layer"},
    )
    pos_norm: bool = field(
        default=False,
        metadata={"help": "whether to use layer norm in position layer"},
    )
    pool_gelu: bool = field(
        default=True,
        metadata={"help": "whether to use gelu in pooling layer"},
    )
    pos_extra_norm: bool = field(
        default=False,
        metadata={"help": "whether to use gelu in pooling layer"},
    )


@register_model("wav2vec2", dataclass=Wav2Vec2Config)
class Wav2Vec2Model(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2Config):
        # breakpoint()
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.latent_vars,
                    temp=cfg.latent_temp,
                    groups=cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )
        # print("!"*10)
        # print("cfg: ", cfg)
        # breakpoint()
        self.encoder = TransformerEncoder(
            cfg)  # TODO: just overwrite this part in the adapted transformer implementation
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Config, task=None):
        """Build a new model instance."""
        # breakpoint()
        return cls(cfg)

    def apply_mask(
            self, x, padding_mask,
            mask_indices=None, mask_channel_indices=None,
    ):
        B, T, C = x.shape
        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                        .to(x.device)
                        .unsqueeze(1)
                        .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def sample_negatives(self, y, num, padding_count=None):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                        .unsqueeze(-1)
                        .expand(-1, self.n_negatives)
                        .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                        .unsqueeze(-1)
                        .expand(-1, self.cross_sample_negatives)
                        .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits = logits / self.logit_temp

        if is_xla_tensor(logits) or neg_is_pos.any():
            fillval = -float(2 ** 30)
            if not hasattr(self, '_inftensor'):
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits) else
                    float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2])

        return input_lengths.to(torch.long)

    def forward(
            self, source, padding_mask=None, mask=True, features_only=False,
            mask_indices=None, mask_channel_indices=None,
            padding_count=None, prior_encoder_features=False,
            n_encoder_layer_features=-1,
            sample_signals=False
    ):
        # breakpoint()
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        # breakpoint()
        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        # breakpoint()
        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[(torch.arange(padding_mask.shape[0], device=padding_mask.device), output_lengths - 1)] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()


        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        extractor_features = features.clone()

        features = self.dropout_input(features)

        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        # breakpoint()
        if not prior_encoder_features:
            x, layer_results = self.encoder(x, padding_mask=padding_mask,
                                            n_encoder_layer_features=n_encoder_layer_features)  # TODO: the easiest way is to return 3 items in adapted transformer layer

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "extractor_features": extractor_features,
                "features": unmasked_features,
                "layer_results": layer_results,
            }

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands = self.quantizer(
                    unmasked_features, produce_targets=False
                )["x"]
                negs, _ = self.sample_negatives(
                    neg_cands, y.size(1), padding_count=padding_count,
                )
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(
                    y, y.size(1), padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features, y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y, y.size(1), padding_count=padding_count,
                )

        if not is_xla_tensor(x):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {
            "x": x, "padding_mask": padding_mask, "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False, prior_encoder_features=False,
                         n_encoder_layer_features=-1, sample_signals=False):
        res = self.forward(source, padding_mask, mask=mask, features_only=True,
                           prior_encoder_features=prior_encoder_features,
                           n_encoder_layer_features=n_encoder_layer_features,
                           sample_signals=sample_signals)
        return res

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None


#
@dataclass
class Wav2Vec2FastConfig(Wav2Vec2Config):
    pass



@register_model("wav2vec2_fast", dataclass=Wav2Vec2FastConfig)
class Wav2Vec2FastModel(Wav2Vec2Model):
    def __init__(self, cfg: Wav2Vec2FastConfig):
        # breakpoint()
        super().__init__(cfg)
        self.encoder = TransformerFastEncoder(cfg)
        # TODO reset encoder layers.

    def forward(
            self, source, padding_mask=None, mask=True, features_only=False,
            mask_indices=None, mask_channel_indices=None,
            padding_count=None, prior_encoder_features=False,
            n_encoder_layer_features=-1,
            sample_signals=False
    ):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        # breakpoint()
        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        # breakpoint()
        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[(torch.arange(padding_mask.shape[0], device=padding_mask.device), output_lengths - 1)] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        extractor_features = features.clone()

        features = self.dropout_input(features)

        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        # breakpoint()
        if not prior_encoder_features:
            x, layer_results, padding_mask = self.encoder(x, padding_mask=padding_mask,
                                                          n_encoder_layer_features=n_encoder_layer_features)  # TODO: the easiest way is to return 3 items in adapted transformer layer

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "extractor_features": extractor_features,
                "features": unmasked_features,
                "layer_results": layer_results,
            }
        print("below code requires modifications if ever used")
        breakpoint()
        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands = self.quantizer(
                    unmasked_features, produce_targets=False
                )["x"]
                negs, _ = self.sample_negatives(
                    neg_cands, y.size(1), padding_count=padding_count,
                )
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(
                    y, y.size(1), padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features, y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y, y.size(1), padding_count=padding_count,
                )

        if not is_xla_tensor(x):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {
            "x": x, "padding_mask": padding_mask, "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def load_state_dict(self, state_dict, strict, model_cfg=None, args=None):
        # breakpoint()
        # print("warning: allowing strict to be false for this version of wav2vec2")
        # state_has_post_extractor_adapter = any([k for k in state_dict.keys() if k.startswith("post_extractor_adapter.")])
        # # breakpoint()
        strict = False
        super().load_state_dict(state_dict, strict, model_cfg, args)

    def reset_encoder_layer_parameters(self, top_layers):
        for i in range(-1, (top_layers - 1), -1):
            print(f"resetting layer {i + 24} parameters in w2v2...")
            self.encoder.layers[i].apply(init_bert_params)


@dataclass
class Wav2Vec2AdaConfig(Wav2Vec2Config):
    pass


@register_model("wav2vec2_in_adapt", dataclass=Wav2Vec2Config)
class Wav2Vec2InAdaModel(Wav2Vec2Model):
    def __init__(self, cfg: Wav2Vec2Config):
        # breakpoint()
        super().__init__(cfg)
        # TODO: overwrite self.encoder to modified encoder layers
        self.encoder = TransformerAdaEncoder(cfg)

    def extract_features(self, source, padding_mask, pass_adapter=False, mask=False, prior_encoder_features=False,
                         n_encoder_layer_features=-1, sample_signals=False):
        res = self.forward(source, padding_mask, pass_adapter=pass_adapter, mask=mask, features_only=True,
                           prior_encoder_features=prior_encoder_features,
                           n_encoder_layer_features=n_encoder_layer_features,
                           sample_signals=sample_signals)
        return res

    def forward(
            self, source, padding_mask=None, pass_adapter=False, mask=True, features_only=False,
            mask_indices=None, mask_channel_indices=None,
            padding_count=None, prior_encoder_features=False,
            n_encoder_layer_features=-1,
            sample_signals=False
    ):
        # print("2"*100)
        # breakpoint()
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        # breakpoint()
        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[(torch.arange(padding_mask.shape[0], device=padding_mask.device), output_lengths - 1)] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        extractor_features = features.clone()

        features = self.dropout_input(features)

        unmasked_features = self.dropout_features(unmasked_features)

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        if not prior_encoder_features:
            x, layer_results, in_adapter_features = self.encoder(x, padding_mask=padding_mask,
                                                                 n_encoder_layer_features=n_encoder_layer_features,
                                                                 pass_adapter=pass_adapter)

        if features_only:
            out = {
                "x": x,
                "padding_mask": padding_mask,
                "extractor_features": extractor_features,
                "features": unmasked_features,
                "layer_results": layer_results,
                "in_adapter_features": in_adapter_features  # when pass_adapter is false, it's pseudo features
            }
            # breakpoint()
            return out

    def load_state_dict(self, state_dict, strict, model_cfg=None, args=None):

        # breakpoint()
        # print("warning: allowing strict to be false for this version of wav2vec2")
        # state_has_post_extractor_adapter = any([k for k in state_dict.keys() if k.startswith("post_extractor_adapter.")])
        # # breakpoint()
        strict = False
        super().load_state_dict(state_dict, strict, model_cfg, args)


@register_model("wav2vec2_out_adapt", dataclass=Wav2Vec2Config)
class Wav2Vec2OutAdaModel(Wav2Vec2Model):
    def __init__(self, cfg: Wav2Vec2Config):
        # breakpoint()
        super().__init__(cfg)
        self.post_extractor_adapter = LightAdapter(
            # arguments to be fixed to be flexable. Now I just make use of default w2v_args that comes with the pretraiend wav2vec model
            cfg.encoder_embed_dim
        )

    def extract_features(self, source, padding_mask, pass_adapter=False, mask=False, prior_encoder_features=False,
                         n_encoder_layer_features=-1, sample_signals=False):
        res = self.forward(source, padding_mask, pass_adapter=pass_adapter, mask=mask, features_only=True,
                           prior_encoder_features=prior_encoder_features,
                           n_encoder_layer_features=n_encoder_layer_features,
                           sample_signals=sample_signals)
        return res

    def forward(
            self, source, padding_mask=None, pass_adapter=False, mask=True, features_only=False,
            mask_indices=None, mask_channel_indices=None,
            padding_count=None, prior_encoder_features=False,
            n_encoder_layer_features=-1,
            sample_signals=False
    ):
        # breakpoint()
        # #print("7" * 50)
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        # breakpoint()
        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[(torch.arange(padding_mask.shape[0], device=padding_mask.device), output_lengths - 1)] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        extractor_features = features.clone()

        features = self.dropout_input(features)

        unmasked_features = self.dropout_features(unmasked_features)

        if self.post_extractor_adapter is not None:
            # breakpoint()
            if pass_adapter:
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! data are passing through post_extractor_adapter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                features, _, _ = self.post_extractor_adapter(features, features)
                post_extractor_adapter_features = features.clone()

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        if not prior_encoder_features:
            x, layer_results = self.encoder(x, padding_mask=padding_mask,
                                            n_encoder_layer_features=n_encoder_layer_features)

        if features_only:
            out = {
                "x": x,
                "padding_mask": padding_mask,
                "extractor_features": extractor_features,
                "features": unmasked_features,
                "layer_results": layer_results,
            }
            if pass_adapter:
                assert self.post_extractor_adapter, "wav2vec must have post_extractor_adapter"
                out["post_extractor_adapter_features"] = post_extractor_adapter_features
            return out

    def load_state_dict(self, state_dict, strict, model_cfg=None, args=None):

        # breakpoint()
        # print("warning: allowing strict to be false for this version of wav2vec2")
        # state_has_post_extractor_adapter = any([k for k in state_dict.keys() if k.startswith("post_extractor_adapter.")])
        # # breakpoint()
        strict = False
        super().load_state_dict(state_dict, strict, model_cfg, args)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
            # sample_signals: False
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        # breakpoint()
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # breakpoint()

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # breakpoint()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, n_encoder_layer_features=-1):
        # breakpoint()
        x, layer_results = self.extract_features(x, padding_mask, n_encoder_layer_features=n_encoder_layer_features)

        if self.layer_norm_first:
            x = self.layer_norm(x)
        # breakpoint()
        return x, layer_results

    def extract_features(self, x, padding_mask=None, n_encoder_layer_features=-1):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2), )
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        # breakpoint()
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                if (n_encoder_layer_features > 0) and (i >= n_encoder_layer_features):
                    break
                if n_encoder_layer_features > -1:
                    print(f"----- getting {i}th layer features -----")
                # breakpoint()
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerFastEncoder(TransformerEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        # breakpoint()

        self.pool_positions = [int(i) for i in cfg.pool_positions.split(",") if i and int(i) > -1]

        if not cfg.extra_pos:
            self.pos_positions = []
        else:
            if cfg.pos_positions:
                self.pos_positions = [int(i) for i in cfg.pos_positions.split(",") if i and int(i) > -1]
            else:
                self.pos_positions = self.pool_positions

        if cfg.share_pos_pool_pos:
            self.pos_positions = self.pool_positions

        if len(self.pool_positions) > 0:
            self.pool_mode = cfg.pool_mode
            pool_layers = [tuple([int(i) for i in cfg.pool_layer.split(",")])] * len(self.pool_positions)

            try:
                assert len(pool_layers) == len(self.pool_positions), "they must be equal"
            except Exception as e:
                print("error: ", e)
                breakpoint()

            self.pool_layers = nn.ModuleList()
            self.pool_op = cfg.pool_op
            for i in range(len(self.pool_positions)):
                dim, k, s = pool_layers[i]
                self.pool_layers.append(
                    PoolLayer(
                        dim=dim,
                        kernel=k,
                        stride=s,
                        mode=cfg.pool_mode,
                        pool_conv_bias=cfg.conv_bias,
                        gelu=cfg.pool_gelu,
                        pool_op=self.pool_op,
                        use_layer_norm=cfg.pool_norm
                    )
                )
        if len(self.pos_positions) > 0:
            dim = self.embedding_dim
            # breakpoint()
            print("adding relative positional information........")
            self.pos_layers = nn.ModuleList()
            self.extra_pos_before_pool = cfg.extra_pos_before_pool
            if cfg.pos_extra_norm:
                print("applying layernorm on pos + x output .....")
                self.pos_extra_norm_layers = nn.ModuleList()
            else:
                self.pos_extra_norm_layers = None
            for i in range(len(self.pos_positions)):
                if cfg.pos_extra_norm:
                    self.pos_extra_norm_layers.append(LayerNorm(self.embedding_dim))
                pos_conv = nn.Conv1d(
                    self.embedding_dim,
                    self.embedding_dim,
                    kernel_size=cfg.conv_pos,
                    padding=cfg.conv_pos // 2,
                    groups=cfg.conv_pos_groups,
                )
                if not cfg.pos_kaiming_init:
                    dropout = 0
                    std = math.sqrt((4 * (1.0 - dropout)) / (cfg.conv_pos * self.embedding_dim))
                    nn.init.normal_(pos_conv.weight, mean=0, std=std)
                    nn.init.constant_(pos_conv.bias, 0)

                    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
                    if not cfg.pos_norm and cfg.pos_gelu:
                        pos_conv = nn.Sequential(pos_conv, SamePad(cfg.conv_pos), nn.GELU())
                    elif not cfg.pos_norm and not cfg.pos_gelu:
                        pos_conv = nn.Sequential(pos_conv, SamePad(cfg.conv_pos))
                    elif cfg.pos_norm and cfg.pos_gelu:
                        print("adding pos norm....")
                        print("use pos gelu....")
                        pos_conv = nn.Sequential(pos_conv,
                                                 nn.Sequential(
                                                     TransposeLast(),
                                                     Fp32LayerNorm(dim, elementwise_affine=True),
                                                     TransposeLast(),
                                                 ),
                                                 SamePad(cfg.conv_pos),
                                                 nn.GELU()
                                                 )
                    elif cfg.pos_norm and not cfg.pos_gelu:
                        print("adding pos norm....")
                        print("no pos gelu.....")
                        pos_conv = nn.Sequential(pos_conv,
                                                 nn.Sequential(
                                                     TransposeLast(),
                                                     Fp32LayerNorm(dim, elementwise_affine=True),
                                                     TransposeLast(),
                                                 ),
                                                 SamePad(cfg.conv_pos),
                                                 )
                else:
                    print("kaiming init for positional layer .....")
                    nn.init.kaiming_normal_(pos_conv.weight)
                    if not cfg.pos_norm and cfg.pos_gelu:
                        pos_conv = nn.Sequential(pos_conv, SamePad(cfg.conv_pos), nn.GELU())
                    elif not cfg.pos_norm and not cfg.pos_gelu:
                        pos_conv = nn.Sequential(pos_conv, SamePad(cfg.conv_pos))
                    elif cfg.pos_norm and cfg.pos_gelu:
                        print("adding pos norm....")
                        print("use  pos gelu.....")
                        pos_conv = nn.Sequential(pos_conv,
                                                 nn.Sequential(
                                                     TransposeLast(),
                                                     Fp32LayerNorm(dim, elementwise_affine=True),
                                                     TransposeLast(),
                                                 ),
                                                 SamePad(cfg.conv_pos),
                                                 nn.GELU()
                                                 )
                    elif cfg.pos_norm and not cfg.pos_gelu:
                        print("adding pos norm....")
                        print("no pos gelu.....")
                        pos_conv = nn.Sequential(pos_conv,
                                                 nn.Sequential(
                                                     TransposeLast(),
                                                     Fp32LayerNorm(dim, elementwise_affine=True),
                                                     TransposeLast(),
                                                 ),
                                                 SamePad(cfg.conv_pos),
                                                 )

                self.pos_layers.append(pos_conv)
        else:
            self.pos_layers = None
            self.extra_pos_before_pool = None

    def forward(self, x, padding_mask=None, n_encoder_layer_features=-1):
        # breakpoint()
        x, layer_results, padding_mask = self.extract_features(x, padding_mask,
                                                               n_encoder_layer_features=n_encoder_layer_features)

        if self.layer_norm_first:
            x = self.layer_norm(x)
        # breakpoint()
        return x, layer_results, padding_mask

    def extract_features(self, x, padding_mask=None, n_encoder_layer_features=-1):
        # change key_padding_mask according to q
        def _get_feat_extract_output_lengths(input_lengths, kernel_size, stride, padding):
            """
            Computes the output length of the convolutional layers
            """

            def _conv_out_length(input_length, kernel_size, stride):
                return torch.floor((input_length + 2 * padding - kernel_size) / stride + 1)

            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
            return input_lengths.to(torch.long)

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        # if not self.training:
        #     breakpoint()
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        # if not self.training:
        #     breakpoint()
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        # if not self.training:
        #     breakpoint()

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        pos_i, pol_i = 0, 0
        for i, layer in enumerate(self.layers):
            # print("i=", i)
            dropout_probability = np.random.random()
            # if i == 15:
            #     breakpoint()

            if not self.training or (dropout_probability > self.layerdrop):
                if len(self.pos_positions) > 0 and i in self.pos_positions and self.extra_pos_before_pool:
                    # print(f"pos position layer: {i}")
                    # breakpoint()
                    x_conv = self.pos_layers[pos_i](x.permute(1, 2, 0))
                    x_conv = x_conv.permute(2, 0, 1)
                    x = x + x_conv

                    pos_i += 1

                if len(self.pool_positions) > 0 and i in self.pool_positions:

                    x = self.pool_layers[pol_i](x)  # TODO: fix padding

                    input_lengths = (1 - padding_mask.long()).sum(-1)
                    # print("input_lengths: ", input_lengths)
                    # breakpoint()

                    if self.pool_op == "conv" or self.pool_op == "mean":
                        k, s, p = self.pool_layers[pol_i].conv_pool[0].kernel_size[0], \
                                  self.pool_layers[pol_i].conv_pool[0].stride[0], \
                                  self.pool_layers[pol_i].conv_pool[0].padding[0]
                    else:
                        k, s, p = self.pool_layers[pol_i].conv_pool[0].kernel_size, \
                                  self.pool_layers[pol_i].conv_pool[0].stride, \
                                  self.pool_layers[pol_i].conv_pool[0].padding
                    try:
                        output_lengths = _get_feat_extract_output_lengths(input_lengths, k, s, p)
                    except Exception as e:
                        print("error: ", e)
                        breakpoint()
                        output_lengths = _get_feat_extract_output_lengths(input_lengths, k, s, p)
                    # breakpoint()
                    assert max(output_lengths) == x.size(
                        0), "Output lengths are wrong!! Errors might be due to rounding!!!"
                    # print("output_lengths: ", output_lengths)
                    # breakpoint()
                    padding_mask = lengths_to_padding_mask(output_lengths)
                    pol_i += 1

                if len(self.pos_positions) > 0 and i in self.pos_positions and not self.extra_pos_before_pool:
                    x_conv = self.pos_layers[pos_i](x.permute(1, 2, 0))
                    x_conv = x_conv.permute(2, 0, 1)
                    x = x + x_conv
                    # breakpoint()
                    if self.pos_extra_norm_layers:
                        x = self.pos_extra_norm_layers[pos_i](x)
                    pos_i += 1

                x, _ = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)

        assert pos_i == len(self.pos_positions)
        assert pol_i == len(self.pool_positions)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        # print(f"!!!! (w2v transformer encoder only) batch_size {x.shape[0]} throughput {(tic2 - tic1)}")

        return x, layer_results, padding_mask


class PoolLayer(nn.Module):
    def __init__(
            self,
            dim: int = 1024,
            kernel: int = 3,
            stride: int = 2,
            dropout: float = 0.0,
            mode: str = "layer_norm",
            pool_conv_bias: bool = False,
            gelu: bool = True,
            pool_op: str = "conv",
            use_layer_norm: bool = True
    ):
        super().__init__()

        assert mode in {"", "group_norm", "layer_norm"}

        def block(
                dim,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                pool_conv_bias=False,
                pool_op="conv",
                gelu=True,
                use_layer_norm=True,
        ):
            def make_pool(op="conv"):
                if op == "conv":
                    print("pooling operation: conv .....")
                    # conv = nn.Conv1d(dim, dim, k, stride=stride, bias=pool_conv_bias)
                    conv = nn.Conv1d(dim, dim, k, stride=stride, bias=pool_conv_bias, padding=k // 2)
                    nn.init.kaiming_normal_(conv.weight)
                    return conv
                elif op == "max":
                    print("pooling operation: max .....")
                    return nn.MaxPool1d(k, stride=stride, padding=k // 2)
                elif op == "mean":
                    print("pooling operation: mean .....")
                    return nn.AvgPool1d(k, stride=stride, padding=k // 2)
                else:
                    raise NotImplementedError

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                modules = nn.Sequential(
                    # TransposeLast(),
                    make_pool(op=pool_op),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast())
                    )

                if gelu:
                    print("using pool gelu ....")
                    modules.add_module("gelu", nn.GELU())
                else:
                    print("no pool gelu ....")
                return modules
            elif is_group_norm:  # TODO: fix dimensions if I want to use it.
                return nn.Sequential(
                    make_pool(),
                    nn.Dropout(p=dropout),
                    # TransposeLast(),
                    Fp32GroupNorm(dim, dim, affine=True),
                    # TransposeLast(),
                    # nn.GELU(),
                )
            else:
                return nn.Sequential(make_pool(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_pool = block(dim,
                               kernel,
                               stride,
                               is_layer_norm=mode == "layer_norm",
                               is_group_norm=mode == "group_norm",
                               pool_conv_bias=pool_conv_bias,
                               pool_op=pool_op,
                               gelu=gelu,
                               use_layer_norm=use_layer_norm
                               )

    def forward(self, x):
        # breakpoint()
        x = x.permute(1, 2, 0)
        x = self.conv_pool(x)
        x = x.permute(2, 0, 1)
        return x


class TransformerAdaEncoder(TransformerEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.ModuleList(
            [
                TransformerAdaSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )

    def forward(self, x, padding_mask=None, n_encoder_layer_features=-1, pass_adapter=False):
        # print("2.1  " * 10)
        x, layer_results, in_adapter_features = self.extract_features(x, padding_mask,
                                                                      n_encoder_layer_features=n_encoder_layer_features,
                                                                      pass_adapter=pass_adapter)

        if self.layer_norm_first:
            x = self.layer_norm(x)
        return x, layer_results, in_adapter_features

    def extract_features(self, x, padding_mask=None, n_encoder_layer_features=-1, pass_adapter=False):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        in_adapter_features = []

        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                if (n_encoder_layer_features > 0) and (i >= n_encoder_layer_features):
                    break
                if n_encoder_layer_features > -1:
                    print(f"----- getting {i}th layer features -----")
                # breakpoint()
                x, _, h = layer(x, self_attn_padding_mask=padding_mask, need_weights=False, pass_adapter=pass_adapter)
                layer_results.append(x)
                in_adapter_features.append(h)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results, in_adapter_features


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "relu",
            layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """

        residual = x


        if self.layer_norm_first:
            #
            x = self.self_attn_layer_norm(x)
            # breakpoint()
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            # #print("6" * 100)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
        # breakpoint()
        return x, attn


class TransformerAdaSentenceEncoderLayer(TransformerSentenceEncoderLayer):
    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "relu",
            layer_norm_first: bool = False,
    ) -> None:

        super().__init__(
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            activation_fn,
            layer_norm_first
        )
        self.adapter = LightAdapter(
            # arguments to be fixed to be flexable. Now I just make use of default w2v_args that comes with the pretraiend wav2vec model
            embedding_dim
        )

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            att_args=None,
            pass_adapter=False
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        # print("4" * 100)
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            if pass_adapter:
                print("!!!!!!!!!!!!!!!! data are passing through in_adapter 000 !!!!!!!!!!!!!!!!!!!!!!!")
                x, _, _ = self.adapter(x, x)
                in_adapter_features = x.clone()
            else:
                in_adapter_features = x.clone()  # pseudo adapter features
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)

            if pass_adapter:
                print("!!!!!!!!!!!!!!!! data are passing through in_extractor_adapter 111 !!!!!!!!!!!!!!!!!!!!!!!")
                x, _, _ = self.adapter(x, x)
                in_adapter_features = x.clone()
            else:
                in_adapter_features = x.clone()  # pseudo adapter features

            x = residual + x
            x = self.final_layer_norm(x)
        return x, attn, in_adapter_features
