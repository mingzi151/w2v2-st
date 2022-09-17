# from fairseq.data.data_utils import compute_mask_indices
import json
import math
import os
import numpy as np
from fairseq.dataclass import ChoiceEnum
from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer

from fairseq_modules.models.wav2vec_s2t import *
from fairseq_modules.modules.multiscale_transformer_layer import MultiScaleEncoderLayer, OutPoolMultiScaleEncoderLayer


@dataclass
class Wav2Vec2Seq2SeqMultiScaleConfig(Wav2Vec2Seq2SeqModConfig):
    multiscale_layers: int = field(
        default=3,
        metadata={"help": "number of multiscale layer in length adaptor."}
    )
    ms_embedding_dim: int = field(
        default=1024, metadata={"help": "encoder embedding dimension"}
    )
    ms_ffn_embedding_dim: int = field(
        default=4096, metadata={"help": "encoder embedding dimension"}
    )
    ms_attention_heads: int = field(
        default=16, metadata={"help": "num encoder attention heads"}
    )
    ms_activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    ms_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for the transformer"}
    )
    ms_attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    ms_activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )


    ms_layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )

    # compressions
    compress_q_kernel_size: str = field(
        default="4", metadata={"help": "kernel size for 1d conv for compressing q."}
    )
    compress_q_stride: str = field(
        default="2", metadata={"help": "stride for 1d conv for compressing q."}
    )
    compress_pad_half_by_kernel: bool = field(
        default=True, metadata={"help": "devide pad by half with kernel or not; if not, devide with stride "}
    )
    compress_k_kernel_size: str = field(
        default="4", metadata={"help": "kernel size for 1d conv for compressing k."}
    )
    compress_k_stride: str = field(
        default="2", metadata={"help": "stride for 1d conv for compressing k."}
    )
    compress_v_kernel_size: str = field(
        default="4", metadata={"help": "kernel size for 1d conv for compressing v."}
    )
    compress_v_stride: str = field(
        default="2", metadata={"help": "stride for 1d conv for compressing v."}
    )

    w2v_encoder_layers: int = field(
        default=-1, metadata={"help": "how many layers of wav2vec 2 transformer encoder to keep."}
    )

    ordinary_trans_layers: int = field(
        default=0, metadata={"help": "use ordinary Transformer layers instead of multiscale transformer"}
    )

    shared_compress_xqkv: bool = field(
        default=False, metadata={"help": "Whether to share pooling module among xqkv at each layer."}
    )
    shared_compress_kv: bool = field(
        default=False, metadata={"help": "Whether to share pooling module among kv at each layer."}
    )
    scales: str = field(
        default="", metadata={"help": "whether to have multi-scales"}
    )

    no_ffn: bool = field(
        default=False, metadata={"help": "Whether to include ffn in M-adapter."}
    )
    use_bart_decoder: bool = field(
        default=False, metadata={"help": "use huggingface bart decoder"}
    )
    use_outpool_multiscale_layer: bool = field(
        default=False, metadata={"help": "use for ablation study"}
    )
    pooling_pos: str = field(
        default="", metadata={"help": " the position of cnn pooling module, select from [start, mid, end]"}
    )
    compress_x_kernel_size: str = field(
        default="3", metadata={"help": "kernel size for 1d conv for compressing x."}
    )
    compress_x_stride: str = field(
        default="2", metadata={"help": "stride for 1d conv for compressing x."}
    )


@register_model("wav2vec_seq2seq_multiscale", dataclass=Wav2Vec2Seq2SeqMultiScaleConfig)
class Wav2Vec2Seq2SeqMultiScaleModel(Wav2Vec2Seq2SeqModModel):
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
    def build_model(cls, cfg: Wav2Vec2Seq2SeqMultiScaleConfig, task: FairseqTask):
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)

        model = Wav2Vec2Seq2SeqMultiScaleModel(encoder, decoder)
        model.freeze_blocks(cfg)
        return model

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2Seq2SeqMultiScaleConfig):
        return Wav2VecEncoderMultiScale(cfg)

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args=None):
        # Allow loading state dicts without Adapter
        # breakpoint()
        state_has_adapter = any([k for k in state_dict.keys() if k.startswith("encoder.adapter.")])
        # state_has_len_adapter = any([k for k in state_dict.keys() if k.startswith("encoder.len_adaptor.")])
        # breakpoint()

        if self.encoder.adapter and not state_has_adapter:
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


class Wav2VecEncoderMultiScale(Wav2VecEncoder):
    """
    Modification of the Wav2VecEncoder

    Similar to Wav2VecEncoderMod, but use Scale Adaptor
    """

    def __init__(self, cfg: Wav2Vec2Seq2SeqMultiScaleConfig, tgt_dict=None):
        super().__init__(cfg, tgt_dict)
        self.remove_w2v_encoder_layers(cfg.w2v_encoder_layers)
        # breakpoint()
        self.adapter = Adapter(
            cfg.decoder_embed_dim,
            cfg.adapter_dim,
            cfg.adapter_dropout
        ) if cfg.adapter_dim else None

        self.multiscale_layers = cfg.multiscale_layers
        self.ordinary_trans_layers = cfg.ordinary_trans_layers
        if self.multiscale_layers > 0:
            assert cfg.ordinary_trans_layers < 1
            self.len_adaptor = self.build_multiscale_adaptor(cfg)
            self.use_outpool_multiscale_layer = cfg.use_outpool_multiscale_layer
        elif self.ordinary_trans_layers > 0:
            assert cfg.multiscale_layers < 1
            self.len_adaptor = self.build_ordinary_trans_adaptor(cfg)
        else:
            raise NotImplementedError
        self.adapter_post = cfg.adapter_post

    def remove_w2v_encoder_layers(self, w2v_encoder_layers):
        if w2v_encoder_layers != -1:
            self.w2v_model.encoder.layers = self.w2v_model.encoder.layers[0:w2v_encoder_layers]

    def build_multiscale_adaptor(self, cfg):
        if not cfg.use_outpool_multiscale_layer:
            return self.build_multiscale_transformer_layer(cfg)
        else:
            assert cfg.pooling_pos != "", "must provide pooling position"
            return self.build_outpool_multiscale_transformer_layer(cfg)

    def build_multiscale_transformer_layer(self, cfg):
        if not cfg.scales:
            return nn.ModuleList(
                MultiScaleEncoderLayer(
                    embedding_dim=cfg.ms_embedding_dim,
                    ffn_embedding_dim=cfg.ms_ffn_embedding_dim,
                    num_attention_heads=cfg.ms_attention_heads,
                    dropout=cfg.ms_dropout,
                    attention_dropout=cfg.ms_attention_dropout,
                    activation_dropout=cfg.ms_activation_dropout,
                    activation_fn=cfg.ms_activation_fn,
                    layer_norm_first=cfg.ms_layer_norm_first,   # in wav2vec config, it is true
                    compress_q_kernel_size=int(cfg.compress_q_kernel_size),
                    compress_q_stride=int(cfg.compress_q_stride),
                    compress_pad_half_by_kernel=cfg.compress_pad_half_by_kernel,
                    compress_k_kernel_size=int(cfg.compress_k_kernel_size),
                    compress_k_stride=int(cfg.compress_k_stride),
                    compress_v_kernel_size=int(cfg.compress_v_kernel_size),
                    compress_v_stride=int(cfg.compress_v_stride),
                    shared_compress_xqkv=cfg.shared_compress_xqkv,
                    shared_compress_kv=cfg.shared_compress_kv,
                    no_ffn=cfg.no_ffn
                )
                for _ in range(cfg.multiscale_layers)
            )
        else:
            print("creating different scales .....")
            scales = [int(i) for i in cfg.scales.split(",")]
            assert sum(scales) == cfg.multiscale_layers, "much be equal"

            def _reformat(param_str):
                param_list = param_str.split(",")
                assert len(param_list) == len(scales)
                param_str = "".join([param_list[i] * scales[i] for i in range(len(param_list))])
                return [int(i) for i in param_str]

            compress_q_kernel_size = _reformat(cfg.compress_q_kernel_size)
            compress_q_stride = _reformat(cfg.compress_q_stride)
            compress_k_kernel_size = _reformat(cfg.compress_k_kernel_size)
            compress_k_stride = _reformat(cfg.compress_k_stride)
            compress_v_kernel_size = _reformat(cfg.compress_v_kernel_size)
            compress_v_stride = _reformat(cfg.compress_v_stride)

            return nn.ModuleList(
                MultiScaleEncoderLayer(
                    embedding_dim=cfg.ms_embedding_dim,
                    ffn_embedding_dim=cfg.ms_ffn_embedding_dim,
                    num_attention_heads=cfg.ms_attention_heads,
                    dropout=cfg.ms_dropout,
                    attention_dropout=cfg.ms_attention_dropout,
                    activation_dropout=cfg.ms_activation_dropout,
                    activation_fn=cfg.ms_activation_fn,
                    layer_norm_first=cfg.ms_layer_norm_first,  # in wav2vec config, it is true
                    compress_q_kernel_size=compress_q_kernel_size[i],
                    compress_q_stride=compress_q_stride[i],
                    compress_pad_half_by_kernel=cfg.compress_pad_half_by_kernel,
                    compress_k_kernel_size=compress_k_kernel_size[i],
                    compress_k_stride=compress_k_stride[i],
                    compress_v_kernel_size=compress_v_kernel_size[i],
                    compress_v_stride=compress_v_stride[i],
                    shared_compress_xqkv=cfg.shared_compress_xqkv,
                    shared_compress_kv=cfg.shared_compress_kv,
                    no_ffn=cfg.no_ffn
                )
                for i in range(len(compress_k_kernel_size))
            )

    def build_outpool_multiscale_transformer_layer(self, cfg):
        if not cfg.scales:
            return nn.ModuleList(
                OutPoolMultiScaleEncoderLayer(
                    embedding_dim=cfg.ms_embedding_dim,
                    ffn_embedding_dim=cfg.ms_ffn_embedding_dim,
                    num_attention_heads=cfg.ms_attention_heads,
                    dropout=cfg.ms_dropout,
                    attention_dropout=cfg.ms_attention_dropout,
                    activation_dropout=cfg.ms_activation_dropout,
                    activation_fn=cfg.ms_activation_fn,
                    layer_norm_first=cfg.ms_layer_norm_first,   # in wav2vec config, it is true
                    compress_x_kernel_size=int(cfg.compress_x_kernel_size),
                    compress_x_stride=int(cfg.compress_x_stride),
                    compress_pad_half_by_kernel=cfg.compress_pad_half_by_kernel,
                    pooling_pos=cfg.pooling_pos
                )
                for _ in range(cfg.multiscale_layers)
            )


    def build_ordinary_trans_adaptor(self, cfg):
        return nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=cfg.ms_embedding_dim,
                    ffn_embedding_dim=cfg.ms_ffn_embedding_dim,
                    num_attention_heads=cfg.ms_attention_heads,
                    dropout=cfg.ms_dropout,
                    attention_dropout=cfg.ms_attention_dropout,
                    activation_dropout=cfg.ms_activation_dropout,
                    activation_fn=cfg.ms_activation_fn,
                    layer_norm_first=cfg.ms_layer_norm_first,
                )
                for _ in range(cfg.ordinary_trans_layers)
            ]
        )


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
        # if not src_lengths[0] == src_lengths[1]:
        # breakpoint()
        # print(src_lengths)

        if not self.training and hasattr(self, 'perturb_ratio') and self.perturb_pos == "input":
            src_tokens = self._perturb_data(src_tokens)

        padding_mask = lengths_to_padding_mask(src_lengths)

        encoder_out = super().forward(
            source=src_tokens,
            padding_mask=padding_mask,
            tbc=False,
            **kwargs
        )   # B x T x D

        encoder_out["encoder_padding_mask"] = encoder_out.pop("padding_mask")

        if not self.training and hasattr(self, 'perturb_ratio') and self.perturb_pos == "w2v_out":
            encoder_out["encoder_out"] = self._perturb_data(encoder_out["encoder_out"])

        if self.adapter and not self.adapter_post:
            encoder_out["encoder_out"] = \
                self.adapter(encoder_out["encoder_out"]
                             )
        x, padding_mask = encoder_out["encoder_out"], encoder_out["encoder_padding_mask"]  # encoder_out["encoder_out"] has the same shape as above
        x = x.permute(1, 0, 2)  # now: T * B * D

        if self.multiscale_layers > 0:
            if not hasattr(self, 'skip_global'):
                for i, layer in enumerate(self.len_adaptor):
                    x, _, padding_mask = layer(x, self_attn_padding_mask=padding_mask)
            elif hasattr(self, 'skip_global'):
                assert self.skip_global == True
                for i, layer in enumerate(self.len_adaptor):
                    x, _, padding_mask = layer(x, self_attn_padding_mask=padding_mask, skip_global=self.skip_global)

        elif self.ordinary_trans_layers > 0:
            for i, layer in enumerate(self.len_adaptor):
                x, _ = layer(x, self_attn_padding_mask=padding_mask)
        else:
            raise NotImplementedError
        # breakpoint()

        encoder_out["encoder_out"] = x  # T * B * D
        encoder_out["encoder_padding_mask"] = padding_mask    # B * T

        if self.adapter and self.adapter_post:
            encoder_out["encoder_out"] = \
                self.adapter(
                    encoder_out["encoder_out"]
                )

        if not self.training and hasattr(self, 'perturb_ratio') and self.perturb_pos == "ada_out":
            # breakpoint()
            x = encoder_out["encoder_out"].permute(1, 0, 2)
            x = self._perturb_data(x)
            encoder_out["encoder_out"] = x.permute(1, 0, 2)

        # breakpoint()
        # print("!!!!!!!!", hasattr(self, "spch_repre_fname"))
        # if hasattr(self, "spch_repre_fname"):
        #     print(self.spch_repre_fname)
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


@dataclass
class Wav2Vec2Seq2SeqMultiScaleRegConfig(Wav2Vec2Seq2SeqMultiScaleConfig, Wav2Vec2Seq2SeqModRegEncoderConfig):
   pass


@register_model("wav2vec_seq2seq_multiscale_reg_enc", dataclass=Wav2Vec2Seq2SeqMultiScaleRegConfig)
class Wav2Vec2Seq2SeqMultiScaleRegModel(Wav2Vec2Seq2SeqRegEncoderModel):
    """
    This model regularises the encoder with a cross-attentive module
    """

    def __init__(self, encoder, decoder, cfg):
        super().__init__(encoder, decoder, cfg)

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

        model = Wav2Vec2Seq2SeqMultiScaleRegModel(encoder, decoder, cfg)
        model.freeze_blocks(cfg)
        return model

    @classmethod
    def build_encoder(cls, cfg):
        return Wav2VecEncoderMultiScale(cfg)
