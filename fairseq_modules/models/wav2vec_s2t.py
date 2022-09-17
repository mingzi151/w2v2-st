import re
import copy
import logging
from typing import Optional
from omegaconf import DictConfig, open_dict
from dataclasses import dataclass, field
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.tasks import FairseqTask
from fairseq.modules import (
    LayerNorm,
    FairseqDropout,
)
from fairseq import checkpoint_utils, utils
from fairseq.models import register_model
from fairseq.models.transformer import TransformerDecoder, TransformerEncoder
from fairseq.models.wav2vec import (
    Wav2Vec2Seq2SeqConfig,
    Wav2Vec2Seq2SeqModel,
    Wav2VecEncoder,
    Wav2VecAdpEncoder,
    Wav2VecFastEncoder,
    Embedding,
)
from fairseq.models.wav2vec import Wav2Vec2OutAdaModel, Wav2Vec2InAdaModel
from fairseq.modules.adapter import LightAdapter

import numpy as np
import os
import json
from omegaconf import MISSING, II
from typing import Optional, Any
from fairseq.models import FairseqDecoder
from fairseq.models.speech_to_text import (

    lengths_to_padding_mask,
    Conv1dSubsampler,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq_modules.models.functions import ReverseLayerF

logger = logging.getLogger(__name__)


def get_avg(last_hiddens, encoder_masks):
    # breakpoint()
    return [(last_hidden.transpose(1, 0) * (~mask).unsqueeze(-1)).sum(1) / (~mask).sum(-1).unsqueeze(-1) for
            (last_hidden, mask) in zip(last_hiddens, encoder_masks)]

def get_first(last_hiddens):
    return last_hiddens[0][0]


def get_enc_pool_out(encoder_out):
    last_hiddens = encoder_out["encoder_out"]
    # enc_pool_out = {
        # "avg": get_avg(last_hiddens, encoder_masks)[0].tolist(),
        #             "first": get_first(last_hiddens).tolist()}
    enc_pool_out = get_first(last_hiddens).tolist()
    return enc_pool_out

def write_enc_to_json(enc_pool_out, spch_repre_fname):
    if os.path.exists(spch_repre_fname):
        with open(spch_repre_fname, "r") as f:
            data = json.load(f)
    else:
        idx = 0
        data = []
    # breakpoint()
    enc_pool_out_dic = {"vector": enc_pool_out}
    data.append(enc_pool_out_dic)
    # data["vectors"].append(enc_pool_out)
    with open(spch_repre_fname, "w") as f:
        json.dump(data, f, indent=2)


BLOCKS2REGEX = {
    "encoder.feat_extr": r"encoder.*\.feature_extractor\..*|"
                         r"encoder.*\.post_extract_proj\..*|"
                         r"encoder.*\.pos_conv\..*",
    "encoder.post_extr_adapter": r"encoder.*\.post_extractor_adapter",
    "encoder.mask_emb": r"encoder.w2v_model.mask_emb",
    "encoder.wav2vec.adapter": r"encoder.w2v_model.*\.adapter\..*",
    "encoder.self_attn": r"encoder.*\.self_attn\..*",
    "encoder.layer_norm": r"encoder.*layer_norm.*",
    "encoder.ffn": r"encoder.*\.fc[1-2]\..*",
    "adapter": r"encoder\.adapter.*",
    "len_adaptor": r"encoder\.len_adaptor.*",
    "post_len_adapter": r"encoder\.post_len_adapter.*",
    "decoder.embedding": r"decoder\.embed_tokens.*|"
                         r"decoder\.embed_positions.*|"
                         r"decoder\.layernorm_embedding.*",
    "decoder.self_attn": r"decoder.*\.self_attn\..*",
    "decoder.layer_norm": r"decoder.*layer_norm.*",
    "decoder.encoder_attn": r"decoder.*\.encoder_attn\..*",
    "decoder.ffn": r"decoder.*\.fc[1-2]\..*",
    "discriminator": r"discriminator.*",
    "text_encoder.embedding": r"text_encoder\.embed_tokens.*|"
                              r"text_encoder\.embed_positions.*|"
                              r"text_encoder\.layernorm_embedding.*",
    "text_encoder.self_attn": r"text_encoder.*\.self_attn\..*",
    "text_encoder.layer_norm": r"text_encoder.*layer_norm.*",
    "text_encoder.ffn": r"text_encoder.*\.fc[1-2]\..*",
}
BLOCKS2REGEX_MOD = {
    "encoder.feat_extr": r"encoder.*\.feature_extractor\..*|"
                         r"encoder.*\.post_extract_proj\..*|"
                         r"encoder.*\.pos_conv\..*",
    "encoder.post_extr_adapter": r"encoder.*\.post_extractor_adapter",
    "encoder.mask_emb": r"encoder.w2v_model.mask_emb",
    "encoder.wav2vec.adapter": r"encoder.w2v_model.*\.adapter\..*",
    "encoder.wav2vec.encoder.pool_layers": r"encoder.w2v_model.encoder.pool_layers\..*",
    "encoder.wav2vec.encoder.pos_layers": r"encoder.w2v_model.encoder.pos_layers\..*",
    "encoder.self_attn": r"encoder.w2v_model.*\.self_attn\..*",
    "encoder.layer_norm": r"encoder.w2v_model.*layer_norm.*",
    "encoder.ffn": r"encoder.w2v_model.*\.fc[1-2]\..*",

    "adapter": r"encoder\.adapter.*",
    "len": r"encoder\.len_adaptor.*",
    "len.self_attn": r"encoder.len_adaptor.*\.self_attn\..*",
    "len.layer_norm": r"encoder.len_adaptor.*layer_norm\..*",
    "len.ffn": r"encoder.len_adaptor.*\.fc[1-2]\..*",
    "len.compress_x": r"encoder.len_adaptor.*\.compress_x\..*",
    "post_len_adapter": r"encoder\.post_len_adapter.*",
    "decoder.embedding": r"decoder\.embed_tokens.*|"
                         r"decoder\.embed_positions.*|"
                         r"decoder\.layernorm_embedding.*",
    "decoder.self_attn": r"decoder.*\.self_attn\..*",
    "decoder.layer_norm": r"decoder.*layer_norm.*",
    "decoder.encoder_attn": r"decoder.*\.encoder_attn\..*",
    "decoder.ffn": r"decoder.*\.fc[1-2]\..*",
    "discriminator": r"discriminator.*",
    "text_encoder.embedding": r"text_encoder\.embed_tokens.*|"
                              r"text_encoder\.embed_positions.*|"
                              r"text_encoder\.layernorm_embedding.*",
    "text_encoder.self_attn": r"text_encoder.*\.self_attn\..*",
    "text_encoder.layer_norm": r"text_encoder.*layer_norm.*",
    "text_encoder.ffn": r"text_encoder.*\.fc[1-2]\..*",
}

# self.encoder.w2v_model.encoder.pos_layers

@dataclass
class Wav2Vec2Seq2SeqModConfig(Wav2Vec2Seq2SeqConfig):
    freeze_layers: str = field(
        default="",
        metadata={"help": "finetune only LayerNorm and Attention (LNA) layers"}
    )
    free_w2v_reset_layers: bool = field(
        default=False,
        metadata={"help": "keep certain w2v enc layer unfrozen."}
    )
    adapter_dim: Optional[int] = field(
        default=None,
        metadata={"help": "projection size of the Adapter"}
    )
    adapter_post: bool = field(
        default=False,
        metadata={"help": "if true, the Adapter is placed after the "
                          "Length Adaptor"}
    )
    adapter_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability for the encoder-decoder "
                          "attention weights (if it's not specified, the "
                          "decoder_attention_dropout is used)"}
    )
    len_adaptor_kernel_sizes: str = field(
        default="3,3",
        metadata={"help": "kernel sizes of the Length Adaptor (Conv1d)"}
    )
    len_adaptor_channels: int = field(
        default=1024,
        metadata={"help": "# of channels in the Length Adaptor (Conv1d)"}
    )
    load_pretrained_decoder_from: Optional[str] = field(
        default=None,
        metadata={"help": "model to take decoder weights from"}
    )
    load_pretrained_encoder_from: Optional[str] = field(
        default=None,
        metadata={"help": "model to take encoder weights from"}
    )
    load_pretrained_text_encoder_from: Optional[str] = field(
        default=None,
        metadata={"help": "load pretrained encoder"}
    )

    decoder_output_dim: int = field(
        default=768,
        metadata={"help": "decoder output dimension (extra linear layer "
                          "if different from decoder embed dim)"}
    )

    decoder_enc_attention_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "dropout probability for the encoder-decoder "
                          "attention weights (if it's not specified, the "
                          "decoder_attention_dropout is used)"}
    )

    discriminator_n_layers: int = field(
        default=3,
        metadata={"help": "Number of feedforward layers in discriminator"}
    )
    discriminator_max_length: int = field(
        default=100,
        metadata={"help": "max length in discriminator"}
    )
    alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "balance seq2seq and discriminator"}
    )

    encoder_is_part_discriminator: bool = field(
        default=False,
        metadata={"help": "whether train encoder with discriminator"}
    )
    freeze_less: bool = field(
        default=True,
        metadata={"help": "Whether to use BLOCKS2REGEX_MOD, which freezes less parameters. "}
    )

    extra_pos: bool = field(
        default=False,
        metadata={"help": "use positional encoding together with pooling layer"},
    )
    extra_pos_before_pool: bool = field(
        default=False,
        metadata={"help": "use positional encoding right before each injected pooling layer"},
    )
    # reset_top_layers: int = field(
    #     default=-1, metadata={"help": "length of the mask for features (channels)"}
    # )
    pos_positions: str = field(
        default="15,19,20",
        metadata={
            "help": "string describing the positions of pooling layers in form of a python list that contains "
                    "[int1, int2, int3]"
        },
    )
    pos_kaiming_init: bool = field(
        default=False,
        metadata={"help": "whether to use kaiming init to initialize position layer"},
    )
    pos_gelu: bool = field(
        default=False,
        metadata={"help": "whether to use gelu in position layer"},
    )
    pos_norm: bool = field(
        default=False,
        metadata={"help": "whether to use layer norm in position layer"},
    )
    pool_positions: str = field(
        default="1,2,3",
        metadata={
            "help": "string describing the positions of pooling layers in form of a python list that contains "
                    "[int1, int2, int3]"
        },
    )
    share_pos_pool_pos: bool = field(
        default=False,
        metadata={"help": "the same positions for pos and pool"},
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
    pool_gelu: bool = field(
        default=False,
        metadata={"help": "whether to use gelu in pooling layer"},
    )
    pos_extra_norm: bool = field(
        default=False,
        metadata={"help": "whether to use gelu in pooling layer"},
    )


@dataclass
class Wav2Vec2Seq2SeqModJointConfig(Wav2Vec2Seq2SeqModConfig):
    text_max_source_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    text_encoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    # text_encoder_output_dim: int = field(
    #     default=768, metadata={"help": "decoder embedding dimension"}
    # )

    text_encoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside the decoder"
        },
    )

    text_encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )

    text_encoder_layers: int = field(default=6, metadata={"help": "num of text encoder layers"})

    text_encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    text_encoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    text_encoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    text_encoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    text_no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    text_encoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )

    text_encoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside the decoder"
        },
    )
    quant_noise_pq_block_size: int = field(
        default=8, metadata={"help": ""}
    )

    lamb: Optional[float] = field(
        default=0,
        metadata={"help": "control the amount of knowledge distilled from text"}
    )







@register_model("wav2vec_seq2seq_iwslt21", dataclass=Wav2Vec2Seq2SeqModConfig)
class Wav2Vec2Seq2SeqModModel(Wav2Vec2Seq2SeqModel):
    """
    Modified version of the wav2vec_seq2seq model.

    It adds these functionalities:
      - Use with the speech_to_text pipeline
      - Loading pretrained decoder
      - Finetuning only LNA layers
      - Using adapter and length_adaptor modules
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModConfig, task: FairseqTask):

        # breakpoint()
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)

        model = Wav2Vec2Seq2SeqModModel(encoder, decoder)
        model.freeze_blocks(cfg)

        return model

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        return Wav2VecEncoderMod(cfg)

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict, embed_tokens):
        decoder = TransformerDecoderMod(cfg, tgt_dict, embed_tokens)
        # breakpoint()
        if getattr(cfg, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=cfg.load_pretrained_decoder_from
            )
            logger.info(
                f"loaded pretrained decoder from: "
                f"{cfg.load_pretrained_decoder_from}"
            )
        return decoder

#    @staticmethod()
    def count_total_params(self, training_only=False):
        if training_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def count_parameters(self):
        try:
            from prettytable import PrettyTable
        except Exception:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
            from prettytable import PrettyTable

        def create_table_and_params(module_name, params_name):
            return PrettyTable([module_name, params_name]), 0

        t_table, t_total_params = create_table_and_params("Modules", "Trainable Parameters")
        u_table, u_total_params = create_table_and_params("Modules", "Total Parameters")

        t_table_w2v_layer0, t_w2v_layer0_params = create_table_and_params("w2v transformer", "Trainable Parameters")
        u_table_w2v_layer0, u_w2v_layer0_params = create_table_and_params("w2v transformer", "Parameters")

        t_table_ms_layer0, t_ms_layer0_params = create_table_and_params("ms len adapter", "Trainable Parameters")
        u_table_ms_layer0, u_ms_layer0_params = create_table_and_params("ms len adapter", "Parameters")

        t_table_cnn_layer0, t_cnn_layer0_params = create_table_and_params("cnn len adapter", "Trainable Parameters")

        t_table_adapter, t_adapter_params = create_table_and_params("adapter", "Trainable Parameters")
        u_table_adapter, u_adapter_params = create_table_and_params("adapter", "Parameters")

        d_t_table, d_t_params = create_table_and_params("decoder", "Trainable Parameters")
        d_u_table, d_u_params = create_table_and_params("decoder", "Parameters")


        for name, parameter in self.named_parameters():
            if name.startswith("decoder"):
                params = parameter.numel()
                d_u_table.add_row([name, params])
                d_u_params += params
                if parameter.requires_grad:
                    d_t_table.add_row([name, params])
                    d_t_params += params
                    if name.startswith("encoder.w2v_model.encoder.layers.0"):
                        d_t_params += params
                        d_t_table.add_row([name, params])
            elif name.startswith("encoder"):
                params = parameter.numel()
                u_table.add_row([name, params])
                u_total_params += params
                if name.startswith("encoder.w2v_model.encoder.layers.0"):
                    u_w2v_layer0_params += params
                    u_table_w2v_layer0.add_row([name, params])
                if name.startswith("encoder.len_adaptor.0"):
                    u_ms_layer0_params += params
                    u_table_ms_layer0.add_row([name, params])
                if name.startswith("encoder.adapter"):
                    u_adapter_params += params
                    u_table_adapter.add_row([name, params])

                if parameter.requires_grad:
                    t_table.add_row([name, params])
                    t_total_params += params
                    if name.startswith("encoder.w2v_model.encoder.layers.0"):
                        t_w2v_layer0_params += params
                        t_table_w2v_layer0.add_row([name, params])
                    if name.startswith("encoder.len_adaptor.0"):
                        t_ms_layer0_params += params
                        t_table_ms_layer0.add_row([name, params])
                    if name.startswith("encoder.len_adaptor.conv_layers.0"):
                        t_cnn_layer0_params += params
                        t_table_cnn_layer0.add_row([name, params])
                    if name.startswith("encoder.adapter"):
                        t_adapter_params += params
                        t_table_adapter.add_row([name, params])
            else:
                continue

        print("############## encoder ###############")
        print(u_table)
        print(t_table)

        print("############## total section ###############")
        print(u_table_w2v_layer0)
        print(u_table_ms_layer0)
        print(u_table_adapter)
        print(f"Total  parames for w2v layer0: {u_w2v_layer0_params:,}")
        print(f"Total  params for ms layer0: {u_ms_layer0_params:,} ")
        print(f"Total  params for adapter: {u_adapter_params:,} ")
        print(f"Total  Params: {u_total_params:,}")

        print("############## trainable section ###############")
        print(t_table_w2v_layer0)
        print(t_table_ms_layer0)
        print(t_table_cnn_layer0)
        print(u_table_adapter)
        print(f"Total trainable parames for w2v layer0: {t_w2v_layer0_params:,}")
        print(f"Total trainable params for ms layer0: {t_ms_layer0_params:,} ")
        print(f"Total trainable params for cnn layer0: {t_cnn_layer0_params:,}")
        print(f"Total trainable params for adapter: {t_adapter_params:,} ")
        print(f"Total trainable Params: {t_total_params:,}")

        print("############## decoder ###############")
        # print(d_t_table)
        # print(d_u_table)
        print(f"Total  parames for decoder: {d_u_params:,}")
        print(f"Total trainable parames for decoder: {d_t_params:,}")

    def forward_throughput(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        import time
        W = 50  # 50
        for i in range(W):
            self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        batch_size = src_tokens.shape[0]
        torch.cuda.synchronize()
        tic1 = time.time()

        N = 20
        for i in range(N):
            encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"!!!! (model encoder only) batch_size {batch_size} throughput {N * batch_size / (tic2 - tic1)}")
        # print(f"!!!! (model encoder only) batch_size {src_tokens.shape[0]} throughput {(tic2 - tic1)}")
        print()

        for key in ["transcript_tokens", "transcript_lengths", "transcript_ntokens"]:
            if key in kwargs:
                del kwargs[key]

        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        torch.cuda.synchronize()
        #    logger.info(f"throughput averaged with 30 times for the encoder only")
        tic3 = time.time()
        print(f"==== (entire model) batch_size {src_tokens.shape[0]} throughput {N * batch_size / (tic3 - tic1)}")
        # breakpoint()
        return decoder_out


    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        #
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # breakpoint()
        # [(k, v[0].size()) for k, v in encoder_out.items() if k != "layer_results"]

        # [('encoder_out', torch.Size([59, 2, 1024])), ('extractor_features', torch.Size([2, 465, 1024])),
         # ('encoder_padding_mask', torch.Size([2, 59]))]
        for key in ["transcript_tokens", "transcript_lengths", "transcript_ntokens"]:
            if key in kwargs:
                del kwargs[key]

        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def get_free_modules(self, cfg):
        free_modules = []
        if cfg.free_w2v_reset_layers:
            N_layers = len(self.encoder.w2v_model.encoder.layers)
            all_layer_modules = ["encoder.w2v_model.encoder.layers." + n for n, _ in self.encoder.w2v_model.encoder.layers.named_parameters()] #[cfg.reset_top_layers:]
            L = len(all_layer_modules)
            free_modules = all_layer_modules[cfg.reset_top_layers*int((L/N_layers)):]
        return free_modules

    def freeze_blocks(self, cfg: Wav2Vec2Seq2SeqModConfig):
        # breakpoint()
        if cfg.freeze_less:
            BLOCK=BLOCKS2REGEX_MOD
        else:
            BLOCK=BLOCKS2REGEX

        free_modules = self.get_free_modules(cfg)
        print("free_modules: ", free_modules)
        if cfg.freeze_layers:
            regex_to_freeze = re.compile(
                "|".join([BLOCK[b] for b in cfg.freeze_layers.split(',')])
            )
            # print("regex_to_freeze: ", regex_to_freeze)

            for n, p in self.named_parameters():
                # print("n:", n)
                # breakpoint()
                if re.match(regex_to_freeze, n) and n not in free_modules:
                    # print("Freezing layer: ")
                    # print(n)
                    p.requires_grad = False
                else:
                    print("tuning layer: ", n)
        # breakpoint()



    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"], sample["label"]

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor if net_output is for translation
        # otherwise, net_output is (B, D) for discriminator
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args=None):
        # Allow loading state dicts without Adapter
        # breakpoint()
        state_has_adapter = any([k for k in state_dict.keys() if k.startswith("encoder.adapter.")])

        if hasattr(self.encoder, "adapter") and self.encoder.adapter and not state_has_adapter:
            state_dict["encoder.adapter.layer_norm.weight"] = \
                self.encoder.adapter.layer_norm.weight
            state_dict["encoder.adapter.layer_norm.bias"] = \
                self.encoder.adapter.layer_norm.bias
            state_dict["encoder.adapter.down_proj.weight"] = \
                self.encoder.adapter.down_proj.weight
            state_dict["encoder.adapter.down_proj.bias"] = \
                self.encoder.adapter.down_proj.bias
            state_dict["encoder.adapter.up_proj.weight"] = \
                self.encoder.adapter.up_proj.weight
            state_dict["encoder.adapter.up_proj.bias"] = \
                self.encoder.adapter.up_proj.bias
        # breakpoint()
        super().load_state_dict(state_dict, strict, model_cfg, args)

    def get_avg(self, last_hiddens, encoder_masks):
        return [(last_hidden.transpose(1, 0) * (~mask).unsqueeze(-1)).sum(1) / (~mask).sum(-1).unsqueeze(-1) \
                for (last_hidden, mask) in zip(last_hiddens, encoder_masks)]


@register_model("wav2vec_seq2seq_iwslt21_pos", dataclass=Wav2Vec2Seq2SeqModConfig)
class Wav2Vec2Seq2SeqModPosModel(Wav2Vec2Seq2SeqModModel):
    """
    Modified version of the wav2vec_seq2seq model.

    It adds these functionalities:
      - Use with the speech_to_text pipeline
      - Loading pretrained decoder
      - Finetuning only LNA layers
      - Using adapter and length_adaptor modules
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModConfig, task: FairseqTask):

        # breakpoint()
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)

        model = Wav2Vec2Seq2SeqModPosModel(encoder, decoder)
        model.freeze_blocks(cfg)

        return model

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        return Wav2VecPosEncoderMod(cfg)


class Wav2VecEncoderMod(Wav2VecEncoder):
    """
    Modification of the Wav2VecEncoder

    It modifies it to work with the speech_to_text pipeline.
    Moreover, it includes the adapter and length adaptor modules.
    """

    def __init__(self, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict=None):
        super().__init__(cfg, tgt_dict)
        self.adapter = Adapter(
            cfg.decoder_embed_dim,
            cfg.adapter_dim,
            cfg.adapter_dropout
        ) if cfg.adapter_dim else None
        # breakpoint()

        self.len_adaptor = Conv1dSubsampler(
            cfg.decoder_embed_dim,
            cfg.len_adaptor_channels,
            cfg.decoder_embed_dim,
            [int(k) for k in cfg.len_adaptor_kernel_sizes.split(",")],
        )
        self.adapter_post = cfg.adapter_post

    def _perturb_data(self, X):
        assert X.size()[0] == 1, "currently only allowing process a single data point"
        X = X.squeeze(0)   # X.shape is B x T x D or B x T
        L = len(X)
        N = math.floor(L*self.perturb_ratio)
        idx = torch.randperm(L)[:N]
        X[idx] = 0
        X = X.unsqueeze(0)
        assert X.size()[0] == 1
        return X

    def forward(self, src_tokens, src_lengths, **kwargs):
        if not self.training and hasattr(self, 'perturb_ratio') and self.perturb_pos == "input":
            src_tokens = self._perturb_data(src_tokens)

        encoder_out = super().forward(
            source=src_tokens,
            padding_mask=lengths_to_padding_mask(src_lengths),
            tbc=False,
            **kwargs
        )  # B x T x D
        #breakpoint()
        [(k, v.size()) for k, v in encoder_out.items() if k != "layer_results"]
        if not self.training and hasattr(self, 'perturb_ratio') and self.perturb_pos == "w2v_out":
            encoder_out["encoder_out"] = self._perturb_data(encoder_out["encoder_out"])

        encoder_out["encoder_padding_mask"] = encoder_out.pop("padding_mask")
        if self.prior_encoder_features or self.n_encoder_layer_features != -1:
            print("------- skipping adaptor and length adaptor ------")
            return {k: [v] for k, v in encoder_out.items()}

        if self.adapter and not self.adapter_post:
            encoder_out["encoder_out"] = \
                self.adapter(encoder_out["encoder_out"]
                             )    # B x T x D
        # breakpoint()
        encoder_out["encoder_out"], lengths = self.len_adaptor(encoder_out["encoder_out"],
                                                               (~encoder_out["encoder_padding_mask"]).sum(dim=1)) # T x B x D

        if self.adapter and self.adapter_post:
            encoder_out["encoder_out"] = \
                self.adapter(
                    encoder_out["encoder_out"]
                )
        encoder_out["encoder_padding_mask"] = lengths_to_padding_mask(lengths)

        if not self.training and hasattr(self, 'perturb_ratio') and self.perturb_pos == "ada_out":
            # breakpoint()
            x = encoder_out["encoder_out"].permute(1, 0, 2)
            x = self._perturb_data(x)
            encoder_out["encoder_out"] = x.permute(1, 0, 2)

        if not self.training and hasattr(self, "spch_repre_fname") and self.spch_repre_fname:
            try:
                print("src_tokens.size():", src_tokens.size())
                enc_pool_out = get_enc_pool_out(encoder_out)
                # enc_pool_out["src_tokens_shape"] = np.array(src_tokens.size()).tolist()
                write_enc_to_json(enc_pool_out, self.spch_repre_fname)
            except Exception as e:
                print("error: ", e)
                breakpoint()
        return {k: [v] for k, v in encoder_out.items()}

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
        }


class TransformerEncoderMod(TransformerEncoder):
    """
    Modification of the TransformerEncoder

    It is adapted to the argument names defined in Wav2Vec2Seq2SeqModConfig.
    """

    def __init__(self, cfg, dictionary, embed_tokens):
        transformer_cfg = copy.deepcopy(cfg)
        # breakpoint()
        with open_dict(transformer_cfg):
            transformer_cfg.dropout = transformer_cfg.text_encoder_dropout
            transformer_cfg.encoder_layerdrop = transformer_cfg.text_encoder_layerdrop
            transformer_cfg.quant_noise_pq_block_size = transformer_cfg.quant_noise_pq_block_size
            transformer_cfg.encoder_learned_pos = transformer_cfg.text_encoder_learned_pos
            transformer_cfg.no_token_positional_embeddings = transformer_cfg.text_no_token_positional_embeddings
            transformer_cfg.encoder_layers = transformer_cfg.text_encoder_layers
            transformer_cfg.encoder_normalize_before = transformer_cfg.text_encoder_normalize_before
            transformer_cfg.encoder_embed_dim = transformer_cfg.text_encoder_embed_dim
            transformer_cfg.attention_dropout = transformer_cfg.text_encoder_attention_dropout
            transformer_cfg.encoder_ffn_embed_dim = transformer_cfg.text_encoder_ffn_embed_dim
            transformer_cfg.encoder_attention_heads = transformer_cfg.text_encoder_attention_heads
            transformer_cfg.activation_dropout = transformer_cfg.text_encoder_activation_dropout
            transformer_cfg.max_source_positions = transformer_cfg.text_max_source_positions

            transformer_cfg.layernorm_embedding = True
            transformer_cfg.adaptive_input = False
            transformer_cfg.no_scale_embedding = False
            transformer_cfg.quant_noise_pq = 0.0
            transformer_cfg.adaptive_softmax_cutoff = None
        super().__init__(transformer_cfg, dictionary, embed_tokens)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)


class TransformerDecoderMod(TransformerDecoder):
    """
    Modification of the TransformerEncoder

    It is adapted to the argument names defined in Wav2Vec2Seq2SeqModConfig.
    """

    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False):
        transformer_cfg = copy.deepcopy(cfg)
        # breakpoint()
        with open_dict(transformer_cfg):
            transformer_cfg.dropout = transformer_cfg.decoder_dropout
            transformer_cfg.attention_dropout = (
                transformer_cfg.decoder_attention_dropout
            )
            transformer_cfg.activation_dropout = (
                transformer_cfg.decoder_activation_dropout
            )

            transformer_cfg.layernorm_embedding = True
            transformer_cfg.adaptive_input = False
            transformer_cfg.no_scale_embedding = False
            transformer_cfg.quant_noise_pq = 0.0
            transformer_cfg.adaptive_softmax_cutoff = None
        super().__init__(transformer_cfg, dictionary, embed_tokens, no_encoder_attn)
        if cfg.decoder_enc_attention_dropout is not None:
            for layer in self.layers:
                layer.encoder_attn.dropout_module.p = \
                    cfg.decoder_enc_attention_dropout

    def load_state_dict(self, state_dict, strict=True):
        # breakpoint()
        state_dict["output_projection.weight"] = state_dict["embed_tokens.weight"]
        super().load_state_dict(state_dict, strict)


class Adapter(nn.Module):
    """
    Adapter for model finetuning, as described in:
    https://arxiv.org/pdf/1909.08478.pdf
    """

    def __init__(self, embed_dim, proj_dim, dropout):
        super().__init__()
        self.layer_norm = LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, proj_dim)
        self.up_proj = nn.Linear(proj_dim, embed_dim)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.up_proj(x)
        x = self.dropout_module(x)
        x += residual
        return x


@dataclass
class Wav2Vec2FastSeq2SeqConfig(Wav2Vec2Seq2SeqModConfig):
    wav2vec_version: str = field(
        default="wav2vec2_fast", metadata={"help": "which version of wav2vec2 to be used; currently only support wav2vec2_fast or wav2vec2"}
    )


@register_model("wav2vec_seq2seq_fast", dataclass=Wav2Vec2FastSeq2SeqConfig)
class Wav2Vec2FastSeq2SeqModel(Wav2Vec2Seq2SeqModModel):
    """
    This model regularises the encoder with a cross-attentive module
    """

    def __init__(self, encoder, decoder, cfg):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2FastSeq2SeqConfig, task: FairseqTask):
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)
        # breakpoint()

        model = Wav2Vec2FastSeq2SeqModel(encoder, decoder, cfg)

        model.freeze_blocks(cfg)
        return model

    @classmethod
    def build_encoder(cls, cfg):
        # if cfg.wav2vec_version == "wav2vec2_fast":
        return Wav2VecFastEncoderMod(cfg)
        # return Wav2VecEncoderMod(cfg)

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args=None):
        # Allow loading state dicts without pooling layers
        # breakpoint()

        # assert model_cfg["wav2vec_version"] == "wav2vec2_fast"
        state_has_in_pool = any([k for k in state_dict.keys() if k.startswith("encoder.w2v_model.encoder.pool_layers.")])
        state_has_in_pos = any([k for k in state_dict.keys() if k.startswith("encoder.w2v_model.encoder.pos_layers.")])
        if hasattr(self.encoder.w2v_model.encoder, "pool_layers") and self.encoder.w2v_model.encoder.pool_layers and not state_has_in_pool:
            N = len(self.encoder.w2v_model.encoder.pool_layers)
            for i in range(0, N):
                state_dict[f"encoder.w2v_model.encoder.pool_layers.{i}.conv_pool.0.weight"] = \
                    self.encoder.w2v_model.encoder.pool_layers[i].conv_pool[0].weight

                state_dict[f"encoder.w2v_model.encoder.pool_layers.{i}.conv_pool.0.bias"] = \
                    self.encoder.w2v_model.encoder.pool_layers[i].conv_pool[0].bias

                state_dict[f"encoder.w2v_model.encoder.pool_layers.{i}.conv_pool.2.0.weight"] = \
                    self.encoder.w2v_model.encoder.pool_layers[i].conv_pool[2][0].weight

                state_dict[f"encoder.w2v_model.encoder.pool_layers.{i}.conv_pool.2.0.bias"] = \
                    self.encoder.w2v_model.encoder.pool_layers[i].conv_pool[2][0].bias

        if hasattr(self.encoder.w2v_model.encoder, "pos_layers") and self.encoder.w2v_model.encoder.pos_layers and not state_has_in_pos:
            N = len(self.encoder.w2v_model.encoder.pos_layers)
            for i in range(0, N):
                state_dict[f"encoder.w2v_model.encoder.pos_layers.{i}.0.bias"] = \
                    self.encoder.w2v_model.encoder.pos_layers[i][0].bias

                state_dict[f"encoder.w2v_model.encoder.pos_layers.{i}.0.weight_g"] = \
                    self.encoder.w2v_model.encoder.pos_layers[i][0].weight_g

                state_dict[f"encoder.w2v_model.encoder.pos_layers.{i}.0.weight_v"] = \
                    self.encoder.w2v_model.encoder.pos_layers[i][0].weight_v

        super().load_state_dict(state_dict, strict, model_cfg, args)

        # model.encoder.w2v_model.encoder.pos_layers[i][0].weight_g


class Wav2VecFastEncoderMod(Wav2VecFastEncoder):
    """
    Modification of the Wav2VecEncoder

    It modifies it to work with the speech_to_text pipeline.
    Moreover, it includes the adapter and length adaptor modules.
    """

    def __init__(self, cfg, tgt_dict=None):
        super().__init__(cfg, tgt_dict)

    def forward(self, src_tokens, src_lengths, **kwargs):
        # breakpoint()
        if not self.training and hasattr(self, 'perturb_ratio') and self.perturb_pos == "input":
            src_tokens = self._perturb_data(src_tokens)

        encoder_out = super().forward(
            source=src_tokens,
            padding_mask=lengths_to_padding_mask(src_lengths),
            tbc=False,
            **kwargs
        )
        # breakpoint()
        encoder_out["encoder_out"] = encoder_out["encoder_out"].permute(1, 0, 2)
        encoder_out['encoder_padding_mask'] = encoder_out['padding_mask']
        assert encoder_out['encoder_padding_mask'].size()[0] == encoder_out["encoder_out"].size()[1]

        return {k: [v] for k, v in encoder_out.items()}

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
        }


class Wav2VecPosEncoderMod(Wav2VecFastEncoderMod):
    """
    Modification of the Wav2VecEncoder

    It modifies it to work with the speech_to_text pipeline.
    Moreover, it includes the adapter and length adaptor modules.
    """

    def __init__(self, cfg, tgt_dict=None):
        super().__init__(cfg, tgt_dict)
        self.adapter = Adapter(
            cfg.decoder_embed_dim,
            cfg.adapter_dim,
            cfg.adapter_dropout
        ) if cfg.adapter_dim else None
        # breakpoint()

        self.len_adaptor = Conv1dSubsampler(
            cfg.decoder_embed_dim,
            cfg.len_adaptor_channels,
            cfg.decoder_embed_dim,
            [int(k) for k in cfg.len_adaptor_kernel_sizes.split(",")],
        )
        self.adapter_post = cfg.adapter_post

    def forward(self, src_tokens, src_lengths, **kwargs):
        if not self.training and hasattr(self, 'perturb_ratio') and self.perturb_pos == "input":
            src_tokens = self._perturb_data(src_tokens)

        encoder_out = super().forward(src_tokens=src_tokens,src_lengths=src_lengths,**kwargs)

        encoder_out = {k: v[0] for k, v in encoder_out.items()}
        encoder_out["encoder_out"] = encoder_out["encoder_out"].permute(1, 0, 2)
        # encoder_out["padding_mask"] = encoder_out["padding_mask"].permute(1, 0)

        # [('encoder_out', torch.Size([2, 465, 1024])), ('extractor_features', torch.Size([2, 465, 1024])),
        #  ('encoder_padding_mask', torch.Size([465, 2])), ('padding_mask', torch.Size([2, 465]))]
        # breakpoint()
        if not self.training and hasattr(self, 'perturb_ratio') and self.perturb_pos == "w2v_out":
            encoder_out["encoder_out"] = self._perturb_data(encoder_out["encoder_out"])


        encoder_out["encoder_padding_mask"] = encoder_out.pop("padding_mask")
        if self.prior_encoder_features or self.n_encoder_layer_features != -1:
            print("------- skipping adaptor and length adaptor ------")
            return {k: [v] for k, v in encoder_out.items()}

        if self.adapter and not self.adapter_post:
            encoder_out["encoder_out"] = \
                self.adapter(encoder_out["encoder_out"]
                             )  # B x T x D

        encoder_out["encoder_out"], lengths = self.len_adaptor(encoder_out["encoder_out"],
                                                               (~encoder_out["encoder_padding_mask"]).sum(
                                                                   dim=1))  # T x B x D

        if self.adapter and self.adapter_post:
            encoder_out["encoder_out"] = \
                self.adapter(
                    encoder_out["encoder_out"]
                )
        encoder_out["encoder_padding_mask"] = lengths_to_padding_mask(lengths)

        if not self.training and hasattr(self, 'perturb_ratio') and self.perturb_pos == "ada_out":
            # breakpoint()
            x = encoder_out["encoder_out"].permute(1, 0, 2)
            x = self._perturb_data(x)
            encoder_out["encoder_out"] = x.permute(1, 0, 2)

        if not self.training and hasattr(self, "spch_repre_fname") and self.spch_repre_fname:
            try:
                print("src_tokens.size():", src_tokens.size())
                enc_pool_out = get_enc_pool_out(encoder_out)
                # enc_pool_out["src_tokens_shape"] = np.array(src_tokens.size()).tolist()
                write_enc_to_json(enc_pool_out, self.spch_repre_fname)
            except Exception as e:
                print("error: ", e)
                breakpoint()
        return {k: [v] for k, v in encoder_out.items()}


@dataclass
class Wav2Vec2Seq2SeqModRegConfig(Wav2Vec2Seq2SeqModConfig):
    proj_speech_n_layer: int = field(
        default=0, metadata={"help": "number of layers for speech projection module"}
    )
    speech_proj_dim: int = field(
        default=4096, metadata={"help": "up projection dimension for speech projection module, if not MLP layer"}
    )
    agg_mode: str = field(
        default="first", metadata={"help": "aggregation method to get speech representation"}
    )


@register_model("wav2vec_seq2seq_reg", dataclass=Wav2Vec2Seq2SeqModRegConfig)
class Wav2Vec2Seq2SeqRegModel(Wav2Vec2Seq2SeqModModel):
    """
    This model only regularises aganist a single vector
    """
    def __init__(self, encoder, decoder, agg_mode="avg"):
        super().__init__(encoder, decoder)
        self.agg_mode = agg_mode

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModRegConfig, task: FairseqTask):
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)

        model = Wav2Vec2Seq2SeqRegModel(encoder, decoder, cfg.agg_mode)
        model.freeze_blocks(cfg)
        return model

    @classmethod
    def build_encoder(cls, cfg):
        return Wav2VecEncoderModReg(cfg)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, proj_speech, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # breakpoint()
        enc_pool_out = self.get_enc_pool_out(proj_speech, encoder_out)

        for key in ["transcript_tokens", "transcript_lengths", "transcript_ntokens"]:
            if key in kwargs:
                del kwargs[key]
        # breakpoint()
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out, enc_pool_out

    def get_enc_pool_out(self, proj_speech, encoder_out):
        last_hiddens = encoder_out["encoder_out"]
        encoder_masks = encoder_out["encoder_padding_mask"]
        # breakpoint()

        # get speech representations

        enc_pool_out = None
        if proj_speech:
            if self.agg_mode == "avg" and not self.encoder.proj_speech:
                enc_pool_out = get_avg(last_hiddens, encoder_masks)[0]
            elif self.agg_mode == "avg" and self.encoder.proj_speech:
                pre_enc_pool_out = get_avg(last_hiddens, encoder_masks)[0]
                enc_pool_out = self.encoder.proj_speech(pre_enc_pool_out)
            elif self.agg_mode == "first" and not self.encoder.proj_speech:
                enc_pool_out = get_first(last_hiddens)
            elif self.agg_mode == "first" and self.encoder.proj_speech:
                pre_enc_pool_out = get_first(last_hiddens)
                enc_pool_out = self.encoder.proj_speech(pre_enc_pool_out)
            else:
                NotImplementedError
        elif not proj_speech:
            if self.agg_mode == "avg":
                enc_pool_out = get_avg(last_hiddens, encoder_masks)[0]
            elif self.agg_mode == "first":
                enc_pool_out = get_first(last_hiddens)
        return enc_pool_out

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args=None):
        # Allow loading state dicts without proj speech layer
        if "proj_speech_n_layer" in model_cfg.keys() and model_cfg["proj_speech_n_layer"]:
            state_has_proj_speech_layer = any([k for k in state_dict.keys() if k.startswith("encoder.proj_speech")])
            # breakpoint()
            if not state_has_proj_speech_layer:
                att = dir(self.encoder.proj_speech)
                if "dense" in att:
                    state_dict["encoder.proj_speech.dense.weight"] = \
                        self.encoder.proj_speech.dense.weight
                    state_dict["encoder.proj_speech.dense.bias"] = \
                        self.encoder.proj_speech.dense.bias
                elif "down_proj" in att:
                    state_dict["encoder.proj_speech.down_proj.weight"] = \
                        self.encoder.proj_speech.down_proj.weight
                    state_dict["encoder.proj_speech.down_proj.bias"] = \
                        self.encoder.proj_speech.down_proj.bias
                    state_dict["encoder.proj_speech.up_proj.weight"] = \
                        self.encoder.proj_speech.up_proj.weight
                    state_dict["encoder.proj_speech.up_proj.bias"] = \
                        self.encoder.proj_speech.up_proj.bias

        if "inject_post_len_adapter" in model_cfg.keys() and model_cfg["inject_post_len_adapter"]:
            state_has_post_len_adapter = any([k for k in state_dict.keys() if k.startswith("encoder.post_len_adapter")])
            # breakpoint()
            if not state_has_post_len_adapter:
                print(f".............. Allow loading state dicts of wav2vec without post_len_adapter ......")
                if model_cfg["light_post_len_adapter"]:
                    print("light version....")
                    state_dict["encoder.post_len_adapter.adapter_down.0.weight"] = \
                        self.encoder.post_len_adapter.adapter_down[0].weight
                    state_dict["encoder.post_len_adapter.adapter_down.0.bias"] = \
                        self.encoder.post_len_adapter.adapter_down[0].bias

                    state_dict["encoder.post_len_adapter.adapter_down.1.weight"] = \
                        self.encoder.post_len_adapter.adapter_down[1].weight
                    state_dict["encoder.post_len_adapter.adapter_down.1.bias"] = \
                        self.encoder.post_len_adapter.adapter_down[1].bias

                    state_dict["encoder.post_len_adapter.adapter_up.weight"] = \
                        self.encoder.post_len_adapter.adapter_up.weight
                    state_dict["encoder.post_len_adapter.adapter_up.bias"] = \
                        self.encoder.post_len_adapter.adapter_up.bias
                else:
                    print("heavier version....")
                    state_dict["encoder.post_len_adapter.layer_norm.weight"] = \
                        self.encoder.post_len_adapter.layer_norm.weight
                    state_dict["encoder.post_len_adapter.layer_norm.bias"] = \
                        self.encoder.post_len_adapter.layer_norm.bias
                    state_dict["encoder.post_len_adapter.down_proj.weight"] = \
                        self.encoder.post_len_adapter.down_proj.weight
                    state_dict["encoder.post_len_adapter.down_proj.bias"] = \
                        self.encoder.post_len_adapter.down_proj.bias
                    state_dict["encoder.post_len_adapter.up_proj.weight"] = \
                        self.encoder.post_len_adapter.up_proj.weight
                    state_dict["encoder.post_len_adapter.up_proj.bias"] = \
                        self.encoder.adapter.up_proj.bias

        # breakpoint()
        if "wav2vec_version" in model_cfg.keys() and "inject_wav2vec_adapter" in model_cfg.keys() and model_cfg["inject_wav2vec_adapter"]:
            if model_cfg["wav2vec_version"] == "wav2vec2_out_adapt":
                state_has_post_extractor_adapter = any(
                    [k for k in state_dict.keys() if k.startswith("encoder.w2v_model.post_extractor_adapter.")])
                if not state_has_post_extractor_adapter:
                    print(f".............. Allow loading state dicts of wav2vec without post_extractor_adapter ......")
                    state_dict["encoder.w2v_model.post_extractor_adapter.adapter_down.0.weight"] = \
                        self.encoder.w2v_model.post_extractor_adapter.adapter_down[0].weight
                    state_dict["encoder.w2v_model.post_extractor_adapter.adapter_down.0.bias"] = \
                        self.encoder.w2v_model.post_extractor_adapter.adapter_down[0].bias

                    state_dict["encoder.w2v_model.post_extractor_adapter.adapter_down.1.weight"] = \
                        self.encoder.w2v_model.post_extractor_adapter.adapter_down[1].weight
                    state_dict["encoder.w2v_model.post_extractor_adapter.adapter_down.1.bias"] = \
                        self.encoder.w2v_model.post_extractor_adapter.adapter_down[1].bias

                    state_dict["encoder.w2v_model.post_extractor_adapter.adapter_up.weight"] = \
                        self.encoder.w2v_model.post_extractor_adapter.adapter_up.weight
                    state_dict["encoder.w2v_model.post_extractor_adapter.adapter_up.bias"] = \
                        self.encoder.w2v_model.post_extractor_adapter.adapter_up.bias

            if model_cfg["wav2vec_version"] == "wav2vec2_in_adapt":
                state_has_in_adapter = any(
                    [k for k in state_dict.keys() if k.startswith("encoder.w2v_model.encoder.layers.0.adapter.")])
                if not state_has_in_adapter:
                    for i in range(0, 24):
                        state_dict[f"encoder.w2v_model.encoder.layers.{i}.adapter.adapter_down.0.weight"] = \
                            self.encoder.w2v_model.encoder.layers[i].adapter.adapter_down[0].weight
                        state_dict[f"encoder.w2v_model.encoder.layers.{i}.adapter.adapter_down.0.bias"] = \
                            self.encoder.w2v_model.encoder.layers[i].adapter.adapter_down[0].bias

                        state_dict[f"encoder.w2v_model.encoder.layers.{i}.adapter.adapter_down.1.weight"] = \
                            self.encoder.w2v_model.encoder.layers[i].adapter.adapter_down[1].weight
                        state_dict[f"encoder.w2v_model.encoder.layers.{i}.adapter.adapter_down.1.bias"] = \
                            self.encoder.w2v_model.encoder.layers[i].adapter.adapter_down[1].bias

                        state_dict[f"encoder.w2v_model.encoder.layers.{i}.adapter.adapter_up.weight"] = \
                            self.encoder.w2v_model.encoder.layers[i].adapter.adapter_up.weight
                        state_dict[f"encoder.w2v_model.encoder.layers.{i}.adapter.adapter_up.bias"] = \
                            self.encoder.w2v_model.encoder.layers[i].adapter.adapter_up.bias

        super().load_state_dict(state_dict, strict, model_cfg, args)





@dataclass
class Wav2Vec2Seq2SeqModRegEncoderConfig(Wav2Vec2Seq2SeqModRegConfig):
    reg_encoder: bool = field(
        default=True, metadata={"help": "whether to regularise the encoder"}
    )
    inject_wav2vec_adapter: bool = field(
        default=False, metadata={"help": "whether to inject wav2vec adapter"}
    )
    inject_post_len_adapter: bool = field(
        default=False, metadata={"help": "whether to inject post len adapter"}
    )
    light_post_len_adapter: bool = field(
        default=True,
        metadata={"help": "whether to use a light version of adapter"}
    )
    wav2vec_version: str = field(
        default="wav2vec2", metadata={"help": "which version of wav2vec2 to be used; wav2vec2 for the original"}
    )
    reg_wav2vec_extractor: bool = field(
        default=False,
        metadata={"help": "Whether to regularize wav2vec feature extractor"}
    )
    reg_wav2vec_adapter: bool = field(
        default=False,
        metadata={"help": "Whether to regularize wav2vec post extractor adapter or in adapter"}
    )

    reg_wav2vec_trans_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to regularize transformer encoder"}
    )

    n_trans_encoder_layer: int = field(
        default=23,
        metadata={"help": "regularize all transformer encoder layers after the n-th layer"}
    )
    reg_wav2vec_trans_encoder_reverse: bool = field(
        default=False,
        metadata={"help": "Whether to regularize transformer encoder from top to bottom."}
    )

    reg_first_adapter: bool = field(
        default=False,
        metadata={"help": "Whether to regularize the 2 adapters"}
    )

    reg_len_adapter: bool = field(
        default=False,
        metadata={"help": "Whether to regularize the length adapter only"}
    )
    reg_post_len_adapter: bool = field(
        default=False,
        metadata={"help": "Whether to regularize the post length adapter"}
    )


@register_model("wav2vec_seq2seq_reg_enc", dataclass=Wav2Vec2Seq2SeqModRegEncoderConfig)
class Wav2Vec2Seq2SeqRegEncoderModel(Wav2Vec2Seq2SeqRegModel):
    """
    This model regularises the encoder with a cross-attentive module
    """

    def __init__(self, encoder, decoder, cfg):
        super().__init__(encoder, decoder, cfg.agg_mode)
        self.reg_encoder = cfg.reg_encoder
        self.inject_wav2vec_adapter = cfg.inject_wav2vec_adapter
        self.inject_post_len_adapter = cfg.inject_post_len_adapter

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModRegEncoderConfig, task: FairseqTask):
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)
        # breakpoint()

        model = Wav2Vec2Seq2SeqRegEncoderModel(encoder, decoder, cfg)
        model.freeze_blocks(cfg)
        return model

    @classmethod
    def build_encoder(cls, cfg):
        if cfg.inject_wav2vec_adapter:
            return Wav2VecAdaEncoderExtraState(cfg)
        return Wav2VecEncoderExtraState(cfg)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, pass_adapter, proj_speech, **kwargs):
        # breakpoint()
        if not self.inject_wav2vec_adapter and not self.inject_post_len_adapter:
            encoder_out = self.encoder(src_tokens, src_lengths=src_lengths,
                                       **kwargs)   # return all features and corresponding mask from speech encoder
        elif self.inject_wav2vec_adapter or self.inject_post_len_adapter:
            # breakpoint()
            # if not self.training:
            #     breakpoint()
            if not self.training and not hasattr(self, "valid"):
                pass_adapter = True
            encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, pass_adapter=pass_adapter, **kwargs)

        for key in ["transcript_tokens", "transcript_lengths", "transcript_ntokens"]:
            if key in kwargs:
                del kwargs[key]
        # breakpoint()
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out, encoder_out


@dataclass
class Wav2Vec2Seq2SeqModRegEncoderParaOptConfig(Wav2Vec2Seq2SeqModRegEncoderConfig):
    reg_group_keys: str = field(
        default="",
        metadata={"help": "specific multiple reg group keys"}
    )


@register_model("wav2vec_seq2seq_reg_enc_para_opt", dataclass=Wav2Vec2Seq2SeqModRegEncoderParaOptConfig)
class Wav2Vec2Seq2SeqModRegEncoderParaOptModel(Wav2Vec2Seq2SeqRegEncoderModel):

    def __init__(self, encoder, decoder, cfg, para_groups):
        super().__init__(encoder, decoder, cfg)
        self.para_groups = para_groups

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModRegEncoderParaOptConfig, task: FairseqTask):
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        def group_parameters(parameters, reg_group):
            regex_to_reg_group = re.compile("|".join([BLOCKS2REGEX[b] for b in reg_group]))
            for n, p in parameters:
                # breakpoint()
                # print("n: ", n)

                if re.match(regex_to_reg_group, n):
                    # print("0")
                    p.param_group = groups[1]
                else:
                    # print("1")
                    p.param_group = groups[0]
            # breakpoint()

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)

        groups = ["main", "reg"]

        model = Wav2Vec2Seq2SeqModRegEncoderParaOptModel(encoder, decoder, cfg, para_groups=groups)
        # breakpoint()
        if not cfg.reg_group_keys:
            reg_group_keys = []
            if cfg.reg_wav2vec_adapter:
                assert cfg.inject_wav2vec_adapter, "cfg.inject_wav2vec_adapter must be true here. "
                if cfg.wav2vec_version == "wav2vec2_out_adapt":
                    reg_group_keys.append("encoder.post_extr_adapter")
                elif cfg.wav2vec_version == "wav2vec2_in_adapt":
                    reg_group_keys.append("encoder.wav2vec.adapter")
                else:
                    raise NotImplementedError
            elif not cfg.inject_wav2vec_adapter:
                if cfg.reg_wav2vec_extractor:
                    assert model.encoder.w2v_model.feature_grad_mult > 0, "feature_grad_mult must be greater than 0 to regularize feature extractor"
                    reg_group_keys.append("encoder.feat_extr")
                if cfg.reg_wav2vec_trans_encoder:
                    reg_group_keys.extend([
                        "encoder.mask_emb",
                        "encoder.self_attn",
                        "encoder.layer_norm",
                        "encoder.ffn"
                    ])
                if cfg.reg_first_adapter:
                    reg_group_keys.append("adapter")
                if cfg.reg_len_adapter:
                    reg_group_keys.append("len_adaptor")
                if cfg.reg_post_len_adapter:
                    assert cfg.inject_post_len_adapter, "cfg.inject_post_len_adapter must be true here. "
                    reg_group_keys.append("post_len_adapter")


        else:
            reg_group_keys = [cfg.reg_group_keys.split(",")]
        print("reg_group_keys: ", reg_group_keys)
        # breakpoint()
        group_parameters(model.named_parameters(), reg_group_keys)

        param_groups = [(p[0], p[1].param_group) for p in model.named_parameters()]
        print("param_groups:", param_groups)

        model.freeze_blocks(cfg)
        return model

    def get_groups_for_update(self, update_num):
        return self.para_groups[0], self.para_groups[1]


class SpeechPrejLayer(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, proj_dim)
        self.up_proj = nn.Linear(proj_dim, embed_dim)

    def forward(self, x):
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.up_proj(x)
        return x


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Wav2VecEncoderModReg(Wav2VecEncoderMod):
    def __init__(self, cfg: Wav2Vec2Seq2SeqModRegConfig, tgt_dict=None):
        super().__init__(cfg, tgt_dict)

        if cfg.proj_speech_n_layer == 0:
            self.proj_speech = None
        elif cfg.proj_speech_n_layer == 1:
            self.proj_speech = MLPLayer(cfg.decoder_embed_dim, cfg.decoder_embed_dim)
        elif cfg.proj_speech_n_layer == 2:
            self.proj_speech = SpeechPrejLayer(cfg.decoder_embed_dim, cfg.speech_proj_dim)
        else:
            NotImplemented

        if cfg._name != "wav2vec_seq2seq_reg":
            if cfg.inject_post_len_adapter:
                if cfg.light_post_len_adapter:
                    self.post_len_adapter = LightAdapter(cfg.decoder_embed_dim)
                else:
                    self.post_len_adapter = Adapter(
                                                    cfg.decoder_embed_dim,
                                                    cfg.adapter_dim,
                                                    cfg.adapter_dropout
                                            ) if cfg.adapter_dim else None

    def forward(self, src_tokens, src_lengths, **kwargs):
        encoder_out = super().forward(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            **kwargs
        )
        return encoder_out


class Wav2VecEncoderExtraState(Wav2VecEncoderModReg):
    """
    Modification of the Wav2VecEncoder

    It modifies it to work with the speech_to_text pipeline.
    Moreover, it includes the adapter and length adaptor modules.
    """

    def __init__(self, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict=None):
        super().__init__(cfg, tgt_dict)

    def forward(self, src_tokens, src_lengths, pass_adapter=False, **kwargs):
        # breakpoint()
        encoder_out = Wav2VecEncoder.forward(
            self,
            source=src_tokens,
            padding_mask=lengths_to_padding_mask(src_lengths),
            tbc=False,
            **kwargs
        )  # encoder_out["encoder_out"] is different from layer_results[-], as it applied with layer_norm

        encoder_out["extractor_features"] = encoder_out["extractor_features"].transpose(1, 0)
        encoder_out["trans_encoder_layer_results"] = encoder_out.pop("layer_results")
        encoder_out["trans_encoder_padding_mask"] = encoder_out.pop("padding_mask")

        if self.adapter and not self.adapter_post and not hasattr(self, "post_len_adapter"):
            print("!!!!!!!!!! no post_len_adapter !!!!!!")
            # breakpoint()
            encoder_out["adapter_out"] = self.adapter(encoder_out["encoder_out"])

            encoder_out["encoder_out"], lengths = self.len_adaptor(encoder_out["adapter_out"],
                                                                   (~encoder_out["trans_encoder_padding_mask"]).sum(
                                                                       dim=1))
            print("------    lengths    ----- ", lengths)
            # breakpoint()

        elif self.adapter and not self.adapter_post and hasattr(self, "post_len_adapter"):
            print("!!!!!!!!!! there is a post_len_adapter !!!!!!")
            encoder_out["adapter_out"] = self.adapter(encoder_out["encoder_out"])

            encoder_out["len_adapter_out"], lengths = self.len_adaptor(encoder_out["adapter_out"],
                                                                   (~encoder_out["trans_encoder_padding_mask"]).sum(
                                                                       dim=1))
            # if not self.training:
            # breakpoint()
            if not self.training and not hasattr(self, "valid"):
                pass_adapter = True

            if pass_adapter:
                print(" !!!!!!!!!!!!!!!   passing post_len_adapter  !!!!!!!!!!")
                if isinstance(self.post_len_adapter, LightAdapter):
                    encoder_out["encoder_out"], _, _ = self.post_len_adapter(encoder_out["len_adapter_out"],
                                                                   encoder_out["len_adapter_out"])  # TODO: length?? mask?  check the shape of their outputs
                elif isinstance(self.post_len_adapter, Adapter):
                    encoder_out["encoder_out"] = self.post_len_adapter(encoder_out["len_adapter_out"])
            else:
                print(" !!!!!!!!!!!!!!!   not passing post_len_adapter  !!!!!!!!!!")
                # breakpoint()
                encoder_out["encoder_out"] = encoder_out["len_adapter_out"].clone()    # pseudo encoder_out for neutral speech
            # encoder_out["len_adapter_out"]
            # breakpoint()
        else:
            NotImplemented
        encoder_out["encoder_padding_mask"] = lengths_to_padding_mask(
            lengths)

        # breakpoint()
        encoder_out['adapter_out'] = encoder_out['adapter_out'].transpose(1, 0)

        return {k: [v] for k, v in encoder_out.items()}  # all output: T * B * D; all mask: B * T


class Wav2VecAdaEncoderExtraState(Wav2VecAdpEncoder):
    """
    Modification of the Wav2Vec2OutAdaModel

    It modifies it to work with the speech_to_text pipeline.
    Moreover, it includes the adapter and length adaptor modules.
    """

    def __init__(self, cfg: Wav2Vec2Seq2SeqModRegEncoderConfig, tgt_dict=None):
        super().__init__(cfg, tgt_dict)
        self.adapter = Adapter(
            cfg.decoder_embed_dim,
            cfg.adapter_dim,
            cfg.adapter_dropout
        ) if cfg.adapter_dim else None

        self.len_adaptor = Conv1dSubsampler(
            cfg.decoder_embed_dim,
            cfg.len_adaptor_channels,
            cfg.decoder_embed_dim,
            [int(k) for k in cfg.len_adaptor_kernel_sizes.split(",")],
        )
        self.adapter_post = cfg.adapter_post

        if cfg.proj_speech_n_layer == 0:
            self.proj_speech = None
        elif cfg.proj_speech_n_layer == 1:
            self.proj_speech = MLPLayer(cfg.decoder_embed_dim, cfg.decoder_embed_dim)
        elif cfg.proj_speech_n_layer == 2:
            self.proj_speech = SpeechPrejLayer(cfg.decoder_embed_dim, cfg.speech_proj_dim)
        else:
            NotImplemented

    def forward(self, src_tokens, src_lengths, pass_adapter=False, **kwargs):
        # breakpoint()
        encoder_out = super().forward(
            source=src_tokens,
            padding_mask=lengths_to_padding_mask(src_lengths),
            pass_adapter=pass_adapter,
            tbc=False,
            **kwargs
        )  # encoder_out["encoder_out"] is different from layer_results[-], as it applied with layer_norm

        encoder_out["extractor_features"] = encoder_out["extractor_features"].transpose(1, 0)
        # breakpoint()
        if pass_adapter:
            if isinstance(self.w2v_model, Wav2Vec2OutAdaModel):
                encoder_out["post_extractor_adapter_features"] = encoder_out[
                    "post_extractor_adapter_features"].transpose(1, 0)
                assert self.wav2vec_version == "wav2vec2_out_adapt", "Wrong wav2vec type. Check your code!!!!"  # sanity check

        if isinstance(self.w2v_model, Wav2Vec2InAdaModel):
            encoder_out["trans_encoder_layer_adapter_features"] = encoder_out.pop(
                "trans_adapter_features")  # return true & pseudo in_adapter_features
            assert self.wav2vec_version == "wav2vec2_in_adapt", "Wrong wav2vec type. Check your code!!!!"  # sanity check

        encoder_out["trans_encoder_layer_results"] = encoder_out.pop("layer_results")
        encoder_out["trans_encoder_padding_mask"] = encoder_out.pop("padding_mask")

        encoder_out["adapter_out"] = self.adapter(encoder_out["encoder_out"])
        encoder_out["encoder_out"], lengths = self.len_adaptor(encoder_out["adapter_out"],
                                                               (~encoder_out["trans_encoder_padding_mask"]).sum(dim=1))
        encoder_out["encoder_padding_mask"] = lengths_to_padding_mask(lengths)

        # breakpoint()
        encoder_out['adapter_out'] = encoder_out['adapter_out'].transpose(1, 0)

        return {k: [v] for k, v in encoder_out.items()}  # all output: T * B * D; all mask: B * T

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
        }


@dataclass
class Wav2Vec2Seq2SeqContModConfig(Wav2Vec2Seq2SeqModConfig):
    index_memory_size: int = field(
        default=220_000,
        metadata={"help": "The number of data that will be applied with contrastive learning term."}
    )

    negative_size: int = field(
        default=20,
        metadata={"help": "The number of similar and negative examples for each data"}
    )

    contrastive_bs: int = field(
        default=10,
        metadata={"help": "Contrastive batch size. Used to select the number of negative pairs from memory bank."}
    )

    temp: float = field(
        default=1,
        metadata={"help": "temperature for infoNCE"}
    )

    momentum: float = field(
        default=1,
        metadata={"help": "momentum rate for feature memory bank."}
    )


@register_model("wav2vec_seq2seq_contrastive", dataclass=Wav2Vec2Seq2SeqContModConfig)
class Wav2Vec2Seq2SeqContModel(Wav2Vec2Seq2SeqRegModel):

    def __init__(self, encoder, decoder, index_memory, feature_memory, sim, contrastive_bs, device):
        super().__init__(encoder, decoder)
        self.index_memory = index_memory
        self.feature_memory = feature_memory
        self.sim = sim
        self.contrastive_bs = contrastive_bs
        self.device = device

        index_memory_updated = torch.zeros(1)

        self.register_buffer("index_memory_updated", index_memory_updated)
        # self.register_buffer("index_mem", index_memory.memory)
        # self.register_buffer("feature_mem", feature_memory.memory)

        # breakpoint()

        assert self.contrastive_bs <= self.index_memory.shape()[
            1], "contrastive_bs must not be greater than the number of topk most similar examples for each data"

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqContModConfig, task: FairseqTask):
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)

        index_memory = IndexMemory(size=cfg.index_memory_size,
                                   negative_size=cfg.negative_size,
                                   device=device)

        feature_memory = FeatureMemory(size=cfg.index_memory_size,
                                       embedding_size=cfg.decoder_embed_dim,
                                       momentum=cfg.momentum,
                                       device=device)

        sim = Similarity(temp=cfg.temp)

        model = Wav2Vec2Seq2SeqContModel(encoder, decoder, index_memory,
                                         feature_memory, sim,
                                         cfg.contrastive_bs,
                                         device=device)
        model.freeze_blocks(cfg)
        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        last_hiddens = encoder_out["encoder_out"]
        encoder_masks = encoder_out["encoder_padding_mask"]
        enc_pool_out = self.get_avg(last_hiddens, encoder_masks)[0]

        for key in ["transcript_tokens", "transcript_lengths", "transcript_ntokens"]:
            if key in kwargs:
                del kwargs[key]

        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out, enc_pool_out

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args=None):
        # Allow loading state dicts without memory bank
        # breakpoint()
        state_has_memory = any([k for k in state_dict.keys() if k.startswith("index_memory")])
        if self.index_memory and not state_has_memory:
            state_dict["index_memory.index_mem"] = self.index_memory.memory
            state_dict["index_memory_updated"] = self.index_memory_updated
            state_dict["feature_memory.feature_mem"] = self.feature_memory.memory

        super().load_state_dict(state_dict, strict, model_cfg, args)


class Similarity(nn.Module):
    # class Similarity:
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class IndexMemory(nn.Module):
    def __init__(self, size=220_000, negative_size=1024, init_tensor=None, device="gpu"):
        """Memory that works on batches.
        Args:
            size (int, optional): Number of memory entries. Defaults to 128.
            embedding_size (int, optional): Size of stored instances. Defaults to 64.
            init_tensor (torch.FloatTensor, optional): Initializes memorywith given tensor. Defaults to None.
            momentum (float, optional): Updates memory only partially, where 1.0 uses only the new representation whereas 0.0 uses only the old representation. Defaults to 1.0.
        """
        super().__init__()
        self.size = size
        self.negative_size = negative_size
        if init_tensor is not None:
            self.memory = init_tensor
        else:
            self.memory = torch.zeros(
                (self.size, negative_size), requires_grad=False, dtype=torch.int64, device=device)
        # breakpoint()
        self.register_buffer("index_mem", self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]

    def shape(self):
        # breakpoint()
        return self.memory.size()

    def update(self, values, idx):
        self.memory[idx] = values


class FeatureMemory(nn.Module):
    def __init__(self, size=220_000, embedding_size=1024, init_tensor=None, momentum=1.0, device="gpu"):
        """Memory that works on batches.
        Args:
            size (int, optional): Number of memory entries. Defaults to 128.
            embedding_size (int, optional): Size of stored instances. Defaults to 64.
            init_tensor (torch.FloatTensor, optional): Initializes memorywith given tensor. Defaults to None.
            momentum (float, optional): Updates memory only partially, where 1.0 uses only the new representation whereas 0.0 uses only the old representation. Defaults to 1.0.
        """
        super().__init__()
        self.size = size
        self.embedding_size = embedding_size
        self.momentum = momentum
        if init_tensor is not None:
            self.memory = init_tensor
        else:
            self.memory = torch.randn(
                (self.size, embedding_size),
                requires_grad=False,
                device=device)
        self.register_buffer("feature_mem", self.memory)

    def update(self, k, idx):
        """Updates memory at given position
        Args:
            k (torch.FloatTensor): Tensor to insert.
            idx (torch.LongTensor): Where to insert.
        """
        # breakpoint()
        # self.memory[idx] = k * self.momentum + \
        #                    self.memory[idx] * (1 - self.momentum)
        try:
            self.memory[idx, :] = k * self.momentum + self.memory[idx, :] * (1 - self.momentum)
        except Exception as e:
            print("error: ", e)
            breakpoint()
            self.memory[idx, :] = k * self.momentum + self.memory[idx, :] * (1 - self.momentum)

    def continuous_update(self, values, start, n):
        # breakpoint()
        self.memory[start:(start + n)] = values * self.momentum + self.memory[start:(start + n)] * (1 - self.momentum)

    def load(self, name):
        self.memory = torch.load(name)

    def save(self, name):
        torch.save(self.memory, name)

    def data(self, m, batch_size, but_idx=None):  # TODO: this function needs to be re-written, as it's not what I want.
        """Returns m many random memory entries.
        Args:
            m (int): Number of entries to return.
            but_idx ([int], optional): Which indices to ignore. Defaults to None.
        Returns:
            torch.FloatTensor: Memory data.
        """
        idx = np.random.choice(
            np.delete(np.arange(self.size), but_idx) if but_idx is not None else np.arange(self.size), m * batch_size)
        return self.memory[idx].reshape(batch_size, m, -1)

    def __getitem__(self, idx):
        return self.memory[idx]


@register_model("wav2vec_asr_ctc", dataclass=Wav2Vec2Seq2SeqModConfig)
class Wav2VecModCtc(Wav2Vec2Seq2SeqModModel):
    def __init__(self, cfg: Wav2Vec2Seq2SeqModConfig, w2v_encoder):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecModCtc(cfg, task.target_dictionary)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float('-inf')

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


@register_model("wav2vec_joint_st", dataclass=Wav2Vec2Seq2SeqModJointConfig)
class Wav2Vec2JointSeq2Seq(Wav2Vec2Seq2SeqModModel):
    """
    Modified version of the Wav2Vec2Seq2SeqModModel model to distill knowledge from text, in a decreasing manner.
    """

    def __init__(self, encoder, decoder, text_encoder, lamb):
        super().__init__(encoder, decoder)
        self.text_encoder = text_encoder
        self.lamb = lamb

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModConfig, task: FairseqTask):
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(task.tgt_dict,
                                               cfg.decoder_embed_dim)  # Embedding(250054, 1024, padding_idx=1)
        text_encoder_embed_tokens = decoder_embed_tokens
        text_encoder = cls.build_text_encoder(cfg, task.tgt_dict, text_encoder_embed_tokens)

        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)

        encoder = cls.build_encoder(cfg)

        model = Wav2Vec2JointSeq2Seq(encoder, decoder, text_encoder, cfg.lamb)
        model.freeze_blocks(cfg)
        return model

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        encoder = Wav2VecEncoderMod(cfg)
        if getattr(cfg, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=cfg.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{cfg.load_pretrained_encoder_from}"
            )
        return encoder

    @classmethod
    def build_text_encoder(cls, cfg: Wav2Vec2Seq2SeqModConfig, src_dict, embed_tokens):
        # from fairseq.models.roberta import RobertaModel
        # text_encoder = RobertaModel.from_pretrained(
        #     model_name_or_path=cfg.pretrained_text_encoder
        # )
        text_encoder = TransformerEncoderMod(cfg, src_dict, embed_tokens)
        if getattr(cfg, "load_pretrained_text_encoder_from", None):
            text_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=text_encoder, checkpoint=cfg.load_pretrained_text_encoder_from
            )
            logger.info(
                f"loaded pretrained text encoder from: "
                f"{cfg.load_pretrained_text_encoder_from}"
            )
        return text_encoder

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict, embed_tokens):
        decoder = TransformerDecoderMod(cfg, tgt_dict, embed_tokens)
        if getattr(cfg, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=cfg.load_pretrained_decoder_from
            )
            logger.info(
                f"loaded pretrained decoder from: "
                f"{cfg.load_pretrained_decoder_from}"
            )
        return decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        # breakpoint()
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, **kwargs
        )

        if not kwargs["transcript_tokens"]:
            raise

        text_encoder_out = self.text_encoder(
            kwargs["transcript_tokens"], src_lengths=kwargs["transcript_lengths"]
        )

        if self.training:
            print("----     adding text information to speech output       -----")
            last_hiddens = text_encoder_out["encoder_out"]
            encoder_masks = text_encoder_out["encoder_padding_mask"]
            pool_out = self.get_avg(last_hiddens, encoder_masks)
            encoder_out["encoder_out"] = [enc_out + self.lamb * p_out.unsqueeze(0) for enc_out, p_out in
                                          zip(encoder_out["encoder_out"], pool_out)]

        for key in ["transcript_tokens", "transcript_lengths", "transcript_ntokens"]:
            del kwargs[key]

        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out


@register_model("wav2vec_adv_seq2seq_iwslt21", dataclass=Wav2Vec2Seq2SeqModConfig)
class Wav2Vec2AdvSeq2Seq(Wav2Vec2Seq2SeqModModel):
    """
    Modified version of the Wav2Vec2Seq2SeqModModel model, replace decoder with FFN.
    """

    def __init__(self, encoder, decoder, discriminator, alpha):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator
        self.alpha = alpha

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModConfig, task: FairseqTask):
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)
        discriminator = cls.build_discriminator(cfg, task.label_dict)
        for p in discriminator.parameters():
            p.param_group = "discriminator"
        for p in decoder.parameters():
            p.param_group = "translator"
        for p in encoder.parameters():
            p.param_group = "translator"

        model = Wav2Vec2AdvSeq2Seq(encoder, decoder, discriminator, cfg.alpha)
        model.freeze_blocks(cfg)
        # breakpoint()
        return model

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        encoder = Wav2VecEncoderMod(cfg)
        # breakpoint()
        if getattr(cfg, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=cfg.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{cfg.load_pretrained_encoder_from}"
            )
        return encoder

    @classmethod
    def build_discriminator(cls, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict):
        return Discriminator(cfg, tgt_dict)

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict, embed_tokens):
        decoder = TransformerDecoderMod(cfg, tgt_dict, embed_tokens)
        if getattr(cfg, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=cfg.load_pretrained_decoder_from
            )
            logger.info(
                f"loaded pretrained decoder from: "
                f"{cfg.load_pretrained_decoder_from}"
            )
        return decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        # breakpoint()
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, **kwargs
        )

        reverse_encoder_out = {"encoder_out": [ReverseLayerF.apply(encoder_out["encoder_out"][0], self.alpha)]}
        for key in ["transcript_tokens", "transcript_lengths", "transcript_ntokens"]:
            if key in kwargs:
                del kwargs[key]

        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )  # (x, x)

        discriminator_out = self.discriminator(prev_output_tokens, reverse_encoder_out)  # (out, _)
        return decoder_out, discriminator_out  # both are tuples

    def get_discriminator_normalized_probs(self, net_output, log_probs, sample=None):
        # if not isinstance(net_output, tuple) and net_output.dim() == 2: # fix dimension for discriminator in adv training
        #     net_output = torch.unsqueeze(net_output, 1)
        lprobs = self.get_discriminator_normalized_probs_scriptable(net_output, log_probs, sample)
        assert net_output.size() == lprobs.size()
        lprobs.batch_first = True
        return lprobs

    def get_discriminator_normalized_probs_scriptable(self, net_output, log_probs, sample):
        # syntactic sugar for simple models which don't have a decoder
        # (e.g., the classification tutorial)
        logits = net_output.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args=None):
        # Allow loading state dicts without memory bank
        # breakpoint()
        state_has_discriminator = any([k for k in state_dict.keys() if k.startswith("discriminator")])
        if self.discriminator and not state_has_discriminator:
            state_dict["discriminator.layers.0.weight"] = self.discriminator.layers[0].weight
            state_dict["discriminator.layers.0.bias"] = self.discriminator.layers[0].bias
            state_dict["discriminator.layers.2.weight"] = self.discriminator.layers[2].weight
            state_dict["discriminator.layers.2.bias"] = self.discriminator.layers[2].bias
            state_dict["discriminator.out.weight"] = self.discriminator.out.weight
            state_dict["discriminator.out.bias"] = self.discriminator.out.bias

        super().load_state_dict(state_dict, strict, model_cfg, args)


@register_model("wav2vec_discriminator_iwslt21", dataclass=Wav2Vec2Seq2SeqModConfig)
class Wav2Vec2Discriminator(Wav2Vec2Seq2SeqModModel):
    """
    Modified version of the Wav2Vec2Seq2SeqModModel model, replace decoder with FFN.
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModConfig, task: FairseqTask):
        """Build a new model instance."""

        encoder = cls.build_encoder(cfg)
        # breakpoint()
        # ????????? Mich: fix tgt_dict here, or fix tgt_dict in task file
        decoder = cls.build_decoder(cfg, task.label_dict)

        model = Wav2Vec2Discriminator(encoder, decoder)
        model.freeze_blocks(cfg)
        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        # breakpoint()
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, **kwargs
        )
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return (decoder_out,)

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        encoder = Wav2VecEncoderMod(cfg)
        # breakpoint()
        if getattr(cfg, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=cfg.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{cfg.load_pretrained_encoder_from}"
            )
        # breakpoint()
        return encoder

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict):
        return Discriminator(cfg, tgt_dict)


class Discriminator(FairseqDecoder):
    def __init__(self, cfg, dictionary):
        super().__init__(dictionary)

        self.encoder_hidden_size = cfg.decoder_embed_dim  # rename it
        self.hidden_size = cfg.decoder_embed_dim
        self.n_layers = cfg.discriminator_n_layers
        self.max_length = cfg.discriminator_max_length

        if self.n_layers > 0:
            layers = list()
            layers.append(nn.Linear(self.encoder_hidden_size * self.max_length, self.hidden_size))
            layers.append(nn.LeakyReLU())
            for i in range(self.n_layers - 2):
                layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                layers.append(nn.LeakyReLU())
            self.layers = nn.ModuleList(layers)
            self.out = nn.Linear(self.hidden_size, 2)
        else:
            self.out = nn.Linear(self.encoder_hidden_size * self.max_length, 2)

    def forward(self, prev_output_tokens, encoder_out):
        # breakpoint()
        enc = encoder_out["encoder_out"][0]
        max_length = enc.size(0)
        batch_size = enc.size(1)

        output = enc.transpose(0, 1).contiguous().view(batch_size, max_length * self.encoder_hidden_size)
        output = F.pad(output, (0, (self.max_length - max_length) * self.encoder_hidden_size), "constant", 0)
        # S = batch_size, max_length * encoder_hidden_size
        if self.n_layers > 0:
            for i in range(len(self.layers)):
                # print("############## \n layer i: ", i)
                output = self.layers[i](output)
        out = self.out(output)
        return out  # .unsqueeze(1)


@register_model("wav2vec_adv_seq2seq_iwslt21_2opt", dataclass=Wav2Vec2Seq2SeqModConfig)
class Wav2Vec2AdvSeq2Seq2Opt(Wav2Vec2AdvSeq2Seq):
    """
    Modified version of the Wav2Vec2Seq2SeqModModel model, replace decoder with FFN.
    Use of 2 optimizers
    """

    def __init__(self, encoder, decoder, discriminator, alpha):
        super().__init__(encoder, decoder, discriminator, alpha)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModConfig, task: FairseqTask):
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)
        discriminator = cls.build_discriminator(cfg, task.label_dict)
        for p in discriminator.parameters():
            p.param_group = "discriminator"
        for p in decoder.parameters():
            p.param_group = "translator"

        if cfg.encoder_is_part_discriminator:
            print("------------ encoder part of discriminator ----------------")
            for p in encoder.parameters():
                p.param_group = "discriminator"
        else:
            print("------------ encoder part of translator ----------------")
            for p in encoder.parameters():
                p.param_group = "translator"

        model = Wav2Vec2AdvSeq2Seq2Opt(encoder, decoder, discriminator, cfg.alpha)
        model.freeze_blocks(cfg)
        # breakpoint()
        return model

    def get_groups_for_update(self, num_updates):
        return "discriminator" if self.discrim_step(num_updates) else "translator"

    def discrim_step(self, num_updates):
        return num_updates % 2 == 1





