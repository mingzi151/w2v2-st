from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer
from fairseq_modules.modules.multiscale_attention import MultiScaleAttention
from fairseq.modules import MultiheadAttention
import torch
from torch import nn
from fairseq.data.data_utils import lengths_to_padding_mask


class MultiScaleEncoderLayer(TransformerSentenceEncoderLayer):
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
            compress_q_kernel_size: float = 1,
            compress_q_stride: float = 1,
            compress_pad_half_by_kernel: bool = True,
            compress_k_kernel_size: float = 1,
            compress_k_stride: float = 1,
            compress_v_kernel_size: float = 1,
            compress_v_stride: float = 1,
            shared_compress_xqkv: bool = False,
            shared_compress_kv: bool = False,
            no_ffn: bool = False
    ):
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

        self.self_attn = MultiScaleAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            compress_q_kernel_size=compress_q_kernel_size,
            compress_q_stride=compress_q_stride,
            compress_pad_half_by_kernel=compress_pad_half_by_kernel,
            compress_k_kernel_size=compress_k_kernel_size,
            compress_k_stride=compress_k_stride,
            compress_v_kernel_size=compress_v_kernel_size,
            compress_v_stride=compress_v_stride,
            shared_compress_xqkv=shared_compress_xqkv,
            shared_compress_kv=shared_compress_kv,
        )
        if shared_compress_xqkv:
            self.compress_x = self.self_attn.compress_q
        else:

            if compress_pad_half_by_kernel:
                self.compress_x = nn.Conv1d(
                    embedding_dim,
                    embedding_dim,
                    compress_q_kernel_size,
                    stride=compress_q_stride,
                    padding=compress_q_kernel_size // 2,
                )
            else:
                self.compress_x = nn.Conv1d(
                    embedding_dim,
                    embedding_dim,
                    compress_q_kernel_size,
                    stride=compress_q_stride,
                    padding=compress_q_stride // 2,
                )

        self.no_ffn = no_ffn
        if self.no_ffn:
            del self.fc1, self.fc2, self.dropout2, self.dropout3, self.activation_fn

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            att_args=None,
            skip_global: bool = False,
    ):
        if not skip_global:
            if self.no_ffn:
                return self._forward_no_ffn(
                    x=x,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_weights=need_weights,
                    att_args=att_args,
                )
            else:
                return self._forward_ffn(
                    x=x,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_weights=need_weights,
                    att_args=att_args,
                )
        else:
            return self._forward_ffn_less(
                x=x,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                att_args=att_args,
            )

    def _forward_ffn(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            att_args=None,
    ):
        # breakpoint()
        residual = x  # T x B x D

        if self.layer_norm_first:

            x = self.self_attn_layer_norm(x)
            x, attn, padding_masks = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            # breakpoint()
            residual = residual.permute(1, 2, 0).contiguous()  # B x D x T
            residual = (
                self.compress_x(residual)
                    .permute(2, 0, 1)
                    .contiguous()
            )
            try:
                assert residual.size() == x.size()  # T x B x D
            except Exception as e:
                print("error: ", e)
                breakpoint()
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
            x, attn, padding_masks = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)

            residual = residual.permute(1, 2, 0).contiguous()
            residual = (
                self.compress_x(residual)
                    .permute(2, 0, 1)
                    .contiguous()
            )
            try:
                assert residual.size() == x.size()
            except Exception as e:
                print("error: ", e)
                breakpoint()

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
        return x, attn, padding_masks

    def _forward_ffn_less(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            att_args=None,
    ):
        # print("skipping global......")
        residual = x  # T x B x D

        if self.layer_norm_first:

            x = self.self_attn_layer_norm(x)
            x, attn, padding_masks = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            # breakpoint()
            residual = residual.permute(1, 2, 0).contiguous()  # B x D x T
            residual = (
                self.compress_x(residual)
                    .permute(2, 0, 1)
                    .contiguous()
            )
            try:
                assert residual.size() == x.size()  # T x B x D
            except Exception as e:
                print("error: ", e)
                breakpoint()

            x = residual

            # residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            # #print("6" * 100)
            x, attn, padding_masks = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)

            residual = residual.permute(1, 2, 0).contiguous()
            residual = (
                self.compress_x(residual)
                    .permute(2, 0, 1)
                    .contiguous()
            )
            try:
                assert residual.size() == x.size()
            except Exception as e:
                print("error: ", e)
                breakpoint()

            x = residual

            # x = self.self_attn_layer_norm(x)

            # residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
        # breakpoint()
        return x, attn, padding_masks

    def _forward_no_ffn(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            att_args=None,
    ):
        residual = x

        if self.layer_norm_first:

            x = self.self_attn_layer_norm(x)
            x, attn, padding_masks = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            residual = residual.permute(1, 2, 0).contiguous()
            residual = (
                self.compress_x(residual)
                    .permute(2, 0, 1)
                    .contiguous()
            )
            try:
                assert residual.size() == x.size()
            except Exception as e:
                print("error: ", e)
                breakpoint()
            x = residual + x

            # residual = x
            x = self.final_layer_norm(x)
            # x = self.activation_fn(self.fc1(x))
            # x = self.dropout2(x)
            # x = self.fc2(x)
            # x = self.dropout3(x)
            # x = residual + x
        else:
            # #print("6" * 100)
            x, attn, padding_masks = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)

            residual = self.compress_x(residual.permute(1, 2, 0).contiguous())
            residual = (
                self.compress_x(residual)
                    .permute(2, 0, 1)
                    .contiguous()
            )
            try:
                assert residual.size() == x.size()
            except Exception as e:
                print("error: ", e)
                breakpoint()

            x = residual + x

            x = self.self_attn_layer_norm(x)

            # residual = x
            # x = self.activation_fn(self.fc1(x))
            # x = self.dropout2(x)
            # x = self.fc2(x)
            # x = self.dropout3(x)
            # x = residual + x
            x = self.final_layer_norm(x)
        # breakpoint()
        return x, attn, padding_masks


class OutPoolMultiScaleEncoderLayer(TransformerSentenceEncoderLayer):
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
            compress_x_kernel_size: int = 1,
            compress_x_stride: int = 1,
            compress_pad_half_by_kernel: bool = True,
            pooling_pos: str = "start"
    ):
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
        self.compress_x_kernel_size = compress_x_kernel_size
        self.compress_x_stride = compress_x_stride

        self.pooling_pos = pooling_pos
        if compress_pad_half_by_kernel:
            self.compress_x_pad = compress_x_kernel_size // 2
        else:
            self.compress_x_pad = compress_x_stride // 2

        self.compress_x = nn.Conv1d(
            embedding_dim,
            embedding_dim,
            compress_x_kernel_size,
            stride=compress_x_stride,
            padding=compress_x_stride // 2,
        )

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            att_args=None,
    ):
        # if self.pooling_pos == "start":
        return self._forward(
            x=x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_weights=need_weights,
            att_args=att_args
        )

    def _forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            att_args=None,
    ):
        # breakpoint()
        if self.pooling_pos == "start":
            x = self._compress_x(x)  # x: # T x B x D
            self_attn_padding_mask = self.recompute_padding_masks(self_attn_padding_mask, x)

        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)

            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            try:
                assert residual.size() == x.size()
            except Exception as e:
                print("error: ", e)
                breakpoint()
            x = residual + x
            if self.pooling_pos == "mid":
                x = self._compress_x(x)
                self_attn_padding_mask = self.recompute_padding_masks(self_attn_padding_mask, x)

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            # #print("6" * 100)
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)

            residual = self.compress_x(residual.permute(1, 2, 0).contiguous())
            residual = (
                self.compress_x(residual)
                    .permute(2, 0, 1)
                    .contiguous()
            )
            try:
                assert residual.size() == x.size()
            except Exception as e:
                print("error: ", e)
                breakpoint()

            x = residual + x
            if self.pooling_pos == "mid":
                x = self._compress_x(x)
                self_attn_padding_mask = self.recompute_padding_masks(self_attn_padding_mask, x)

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        if self.pooling_pos == "end":
            x = self._compress_x(x)

            self_attn_padding_mask = self.recompute_padding_masks(self_attn_padding_mask, x)
        return x, _, self_attn_padding_mask

    def _compress_x(self, x):
        x = x.permute(1, 2, 0).contiguous()  # B x D x T
        x = (
            self.compress_x(x)
                .permute(2, 0, 1)
                .contiguous()
        )
        return x

    def recompute_padding_masks(self, x_padding_mask, x):
        x = x.permute(1, 0, 2)

        def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
            """
            Computes the output length of the convolutional layers
            """

            def _conv_out_length(input_length, kernel_size, stride):
                return torch.floor((input_length + 2 * self.compress_x_pad - kernel_size) / stride + 1)

            input_lengths = _conv_out_length(input_lengths, self.compress_x_kernel_size, self.compress_x_stride)
            return input_lengths.to(torch.long)

        input_lengths = (1 - x_padding_mask.long()).sum(-1)
        output_lengths = _get_feat_extract_output_lengths(input_lengths)

        assert max(output_lengths) == x.size(1), "Output lengths are wrong!! Errors might be due to rounding!!!"
        new_x_padding_mask = lengths_to_padding_mask(output_lengths)

        return new_x_padding_mask
