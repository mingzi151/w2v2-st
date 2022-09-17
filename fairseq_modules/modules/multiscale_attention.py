import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.data.data_utils import lengths_to_padding_mask


class MultiScaleAttention(MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        compress_q_kernel_size=1,
        compress_q_stride=1,
        compress_pad_half_by_kernel=True,
        # compress_q_pad=1,
        compress_k_kernel_size=1,
        compress_k_stride=1,
        # compress_k_pad=1,
        compress_v_kernel_size=1,
        compress_v_stride=1,
        # compress_v_pad=1,
        shared_compress_xqkv=False,
        shared_compress_kv=False
    ):

        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            self_attention,
            encoder_decoder_attention,
            q_noise,
            qn_block_size,
        )
        self.compress_q_kernel_size = compress_q_kernel_size
        self.compress_q_stride = compress_q_stride
        if compress_pad_half_by_kernel:
            self.compress_q_pad = compress_q_kernel_size // 2
            compress_k_pad = compress_k_kernel_size // 2
            compress_v_pad = compress_v_kernel_size // 2
        else:
            self.compress_q_pad = compress_q_stride // 2
            compress_k_pad = compress_k_stride // 2
            compress_v_pad = compress_v_stride // 2
        # used for compress sequence to subsequence
        self.compress_q = nn.Conv1d(
            embed_dim,
            embed_dim,
            compress_q_kernel_size,
            stride=compress_q_stride,
            padding=self.compress_q_pad,
        )
        self.compress_k = nn.Conv1d(
            embed_dim,
            embed_dim,
            compress_k_kernel_size,
            stride=compress_k_stride,
            padding=compress_k_pad
        )
        self.compress_v = nn.Conv1d(
            embed_dim,
            embed_dim,
            compress_v_kernel_size,
            stride=compress_v_stride,
            padding=compress_v_pad,
        )
        if shared_compress_xqkv:
            self.compress_k = self.compress_q
            self.compress_v = self.compress_q
        elif shared_compress_kv:
            self.compress_v = self.compress_k




    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        # breakpoint()
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, key_embed_dim = key.size()
            if not torch.jit.is_scripting():
                assert (key_bsz, key_embed_dim) == (bsz, embed_dim)
                assert value is not None
                assert (src_len, bsz, embed_dim) == value.shape

        if incremental_state is not None:
            breakpoint()
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            # breakpoint()
            q = self.q_proj(query)  # T x B x D
            k = self.k_proj(query)
            v = self.v_proj(query)

            q = q.permute(1, 2, 0).contiguous()  # B x D x T
            q = (
                self.compress_q(q)
                .permute(2, 0, 1)
                .contiguous()
            )
            k = k.permute(1, 2, 0).contiguous()
            k = (
                self.compress_k(k)
                    .permute(2, 0, 1)
                    .contiguous()
            )
            v = v.permute(1, 2, 0).contiguous()
            v = (
                self.compress_v(v)
                    .permute(2, 0, 1)
                    .contiguous()
            )

        else:
            NotImplementedError

        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            # breakpoint()
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )
        # breakpoint()
        assert q.size(0) <= query.size(0)
        assert k.size(0) <= key.size(0)
        tgt_len, bsz, embed_dim = q.size()
        src_len = tgt_len
        q = (
            q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )

        # change key_padding_mask according to q
        def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
            """
            Computes the output length of the convolutional layers
            """

            def _conv_out_length(input_length, kernel_size, stride):
                return torch.floor((input_length + 2 * self.compress_q_pad - kernel_size) / stride + 1)
            input_lengths = _conv_out_length(input_lengths, self.compress_q_kernel_size, self.compress_q_stride)
            return input_lengths.to(torch.long)
        # breakpoint()
        input_lengths = (1 - key_padding_mask.long()).sum(-1)
        output_lengths = _get_feat_extract_output_lengths(input_lengths)
        # breakpoint()
        assert max(output_lengths) == q.size(1), "Output lengths are wrong!! Errors might be due to rounding!!!"

        # breakpoint()
        new_key_padding_mask = lengths_to_padding_mask(output_lengths)
        try:
            assert max(output_lengths) == q.size(1)
            key_padding_mask = new_key_padding_mask
        except Exception as e:
            print("error: ", e)
            breakpoint()

        src_len = tgt_len = q.size(1)

        assert k is not None
        # assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        # breakpoint()
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        try:
            assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        except Exception as e:
            print("error:", e)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        # breakpoint()
        assert v is not None
        attn = torch.bmm(attn_probs, v)  # attn_probs: num_heads*tgt_len*head_dim;  v: num_heads*tgt_len*head_dim (possibly, len(v)=len(q)-1)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        #breakpoint()
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        # breakpoint()
        return attn, attn_weights, key_padding_mask  #  !!!
