# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn as nn
from fairseq_modules.models.wav2vec_s2t import Wav2Vec2Seq2SeqModRegEncoderParaOptModel
import numpy as np

@dataclass
class ExtraLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    # print("target.size before: ", target.size())
    # # breakpoint()
    # if target.dim() == 2:
    #
    # breakpoint()
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
        # print("target.size after: ", target.size())
    assert target.dim() == lprobs.dim()
    # # breakpoint()
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        # print("target.size(): ", target.size())
        # print("nll_loss before reduce", nll_loss)
        nll_loss = nll_loss.sum()
        # print("*" * 100)
        # print("nll_loss after reduce:", nll_loss)
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)

    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    # print("total loss after smoothing: ", loss)
    # print("*"*100)
    # breakpoint()
    return loss, nll_loss


def discriminator_label_smoothed_nll_loss(lprobs, target, epsilon, reduce=True):
    epsilon = 0
    # print("target.size before: ", target.size())
    # # breakpoint()
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    assert target.dim() == lprobs.dim()
    # breakpoint()
    # exit()
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    del lprobs, target
    return loss, nll_loss


def calculate_expressive_neutral_translation_loss(total_loss, lprobs, target, label, epsilon, ignore_index=None,
                                                  reduce=True):
    """
    total_loss: translation loss
    lprobs: translation probabilities, B * T * D
    target: translations
    label: speech labels
    """

    # assert target.dim() == lprobs.dim()
    with torch.no_grad():
        total_loss = total_loss.detach().clone()
        copy_lprobs = lprobs.detach().clone()

        B = len(label)
        T = int(len(target) / B)
        D = copy_lprobs.size()[-1]

        target = target.view(B, T)  # reshape target
        copy_lprobs = copy_lprobs.view(B, T, D)  # reshape target

        expressive_indices = (label == 1).nonzero(as_tuple=True)[0]  # select indices based on label
        neutral_indices = (label == 0).nonzero(as_tuple=True)[0]

        expressive_loss = cal_one_type_spch_translation_loss(copy_lprobs, target, expressive_indices, epsilon,
                                                             ignore_index, reduce)
        neutral_loss = cal_one_type_spch_translation_loss(copy_lprobs, target, neutral_indices, epsilon, ignore_index,
                                                          reduce)
    # print("~" * 100)
    # print("total expressive_loss translation: ", expressive_loss)
    # print("total neutral_loss translation: ", neutral_loss)
    # if len(expressive_indices) != 0:
    #     print("=> average expressive_loss translation: ", expressive_loss / len(expressive_indices))
    #     print("number of expressive datasets in this batch: ", len(expressive_indices))
    # if len(neutral_indices) != 0:
    #     print("=> average neutral_loss translation: ", neutral_loss / len(neutral_indices))
    #     print("number of neutral datasets in this batch: ", len(neutral_indices))
    #
    # print("~" * 100)
    # breakpoint()

    del total_loss, copy_lprobs
    return expressive_loss, neutral_loss


def cal_one_type_spch_translation_loss(lprobs, target, selected_indices, epsilon, ignore_index=None, reduce=True):
    selected_lprobs = torch.index_select(lprobs, 0, selected_indices)
    selected_target = torch.index_select(target, 0, selected_indices)

    selected_lprobs = selected_lprobs.view(-1, selected_lprobs.size(-1))
    selected_target = selected_target.view(-1)

    one_type_error, _ = label_smoothed_nll_loss(selected_lprobs, selected_target, epsilon, ignore_index, reduce)
    return one_type_error  # .item()


def calculate_expressive_neutral_discriminator_loss(total_loss, lprobs, target, reduce=True):
    # # breakpoint()
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    assert target.dim() == lprobs.dim()
    total_loss = total_loss.detach().clone()
    copy_lprobs = lprobs.detach().clone()

    expressive_indices = (target == 1).nonzero(as_tuple=True)[0]
    selected_lprobs = torch.index_select(copy_lprobs, 0, expressive_indices)
    selected_target = torch.index_select(target, 0, expressive_indices)
    # calculate losses for expressive speeches
    expressive_loss, _ = discriminator_label_smoothed_nll_loss(selected_lprobs, selected_target, epsilon=0, reduce=True)

    # breakpoint()
    neutral_loss = total_loss - expressive_loss
    del total_loss, copy_lprobs
    return expressive_loss, neutral_loss


def compute_ncorrect(lprobs, targets):
    if len(lprobs) == 0:
        return torch.tensor(0, device=lprobs.device), torch.tensor(0, device=lprobs.device)
    if targets.dim() == 2:
        targets = targets.squeeze(-1)

    preds = lprobs.argmax(dim=1)  # sentence prediction code uses the orignal logits; here I use normalised logits.

    ncorrect = (preds == targets).sum()
    incorrect = (preds != targets).sum()
    # breakpoint()
    return ncorrect, incorrect


def calculate_expressive_neutral_discriminator_accuracy(lprobs, target):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    assert target.dim() == lprobs.dim()
    copy_lprobs = lprobs.detach().clone()
    expressive_indices = (target == 1).nonzero(as_tuple=True)[0]
    neutral_indices = (target == 0).nonzero(as_tuple=True)[0]

    selected_exp_lprobs = torch.index_select(copy_lprobs, 0, expressive_indices)
    selected_exp_target = torch.index_select(target, 0, expressive_indices)
    expressive_ncorrect, expressive_incorrect = compute_ncorrect(selected_exp_lprobs, selected_exp_target)
    selected_neu_lprobs = torch.index_select(copy_lprobs, 0, neutral_indices)
    selected_neu_target = torch.index_select(target, 0, neutral_indices)
    neutral_ncorrect, neutral_incorrect = compute_ncorrect(selected_neu_lprobs, selected_neu_target)
    return expressive_ncorrect, expressive_incorrect, neutral_ncorrect, neutral_incorrect


@register_criterion(
    "extra_label_smoothed_cross_entropy", dataclass=ExtraLabelSmoothedCrossEntropyCriterionConfig
)
class ExtraLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.report_expressive_neutral_loss = task.report_expressive_neutral_loss
        self.report_expressive_neutral_accuracy = task.report_expressive_neutral_accuracy
        self.theta = task.theta

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #
        # breakpoint()
        net_output = model(**sample["net_input"])
        # print("net_output: ", net_output)
        # breakpoint()g
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        if not self.report_expressive_neutral_loss:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        else:
            decoder_loss, decoder_nll_loss, decoder_expressive_loss, decoder_neutral_loss = self.compute_decoder_separate_loss(
                model, net_output, sample, reduce=reduce)
            loss = decoder_loss
            nll_loss = decoder_nll_loss
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "expressive_loss": decoder_expressive_loss.data,
                "neutral_loss": decoder_neutral_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        # print("loss: ", loss)
        if torch.isnan(loss):
            print("net_output: ", net_output)

            print("0000  net_output[0].isnan().sum()=", net_output[0].isnan().sum())
            breakpoint()
            net_output = model(**sample["net_input"])
            breakpoint()
            net_output = model(**sample["net_input"])
            print("1111  net_output[0].isnan().sum()=", net_output[0].isnan().sum())
            breakpoint()
            net_output = model(**sample["net_input"])
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)  # the last dimention of lprobs is either ~250k for translation
        # # breakpoint()
        loss, nll_loss = label_smoothed_nll_loss(lprobs, target, self.eps, ignore_index=self.padding_idx,reduce=reduce,)
        # # breakpoint()
        return loss, nll_loss

    def compute_decoder_separate_loss(self, model, net_output, sample, reduce=True):
        # # breakpoint()
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        # breakpoint()
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        # breakpoint()
        label = self.get_label_only(model, net_output, sample)
        expressive_loss, neutral_loss = calculate_expressive_neutral_translation_loss(
            loss,
            lprobs,
            target,
            label,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=True)
        loss, expressive_loss, neutral_loss = loss / self.theta, \
                                              expressive_loss / self.theta, neutral_loss / self.theta
        # assert expressive_loss.requires_grad == False, neutral_loss.requires_grad == False
        return loss, nll_loss, expressive_loss, neutral_loss

    def get_lprobs_and_target(self, model, net_output, sample):
        # breakpoint()
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # # breakpoint()
        target, label = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def get_lprobs_and_label(self, model, net_output, sample):
        # # breakpoint()
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # # breakpoint()
        _, label = model.get_targets(sample, net_output)
        return lprobs, label.view(-1)

    def get_label_only(self, model, net_output, sample):
        _, label = model.get_targets(sample, net_output)
        return label.view(-1)

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        # breakpoint()
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        # breakpoint()
        # for step 1 & 2
        if "expressive_loss" in logging_outputs[0].keys():
            expressive_loss_sum = sum(log.get("expressive_loss", 0) for log in logging_outputs)
            neutral_loss_sum = sum(log.get("neutral_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "expressive_loss", expressive_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "neutral_loss", neutral_loss_sum / sample_size / math.log(2), sample_size, round=3
            )

        # for step 3
        if "decoder_loss" in logging_outputs[0].keys() and "discriminator_loss" in logging_outputs[0].keys():
            decoder_loss_sum = sum(log.get("decoder_loss", 0) for log in logging_outputs)
            decoder_nll_loss_sum = sum(log.get("decoder_nll_loss", 0) for log in logging_outputs)
            discriminator_loss_sum = sum(log.get("discriminator_loss", 0) for log in logging_outputs)
            discriminator_nll_loss_sum = sum(log.get("discriminator_nll_loss", 0) for log in logging_outputs)

            metrics.log_scalar(
                "decoder_loss", decoder_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "discriminator_loss", discriminator_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "decoder_nll_loss", decoder_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_scalar(
                "discriminator_nll_loss", discriminator_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
            )

            if "decoder_expressive_loss" and "discriminator_neutral_loss" in logging_outputs[0].keys():
                decoder_expressive_loss_sum = sum(log.get("decoder_expressive_loss", 0) for log in logging_outputs)
                decoder_neutral_loss_sum = sum(log.get("decoder_neutral_loss", 0) for log in logging_outputs)
                discriminator_expressive_loss_sum = sum(
                    log.get("discriminator_expressive_loss", 0) for log in logging_outputs)
                discriminator_neutral_loss_sum = sum(
                    log.get("discriminator_neutral_loss", 0) for log in logging_outputs)

                metrics.log_scalar(
                    "decoder_expressive_loss", decoder_expressive_loss_sum / sample_size / math.log(2), sample_size,
                    round=3
                )
                metrics.log_scalar(
                    "decoder_neutral_loss", decoder_neutral_loss_sum / sample_size / math.log(2), sample_size, round=3
                )

                metrics.log_scalar(
                    "discriminator_expressive_loss", discriminator_expressive_loss_sum / sample_size / math.log(2),
                    sample_size, round=3
                )

                metrics.log_scalar(
                    "discriminator_neutral_loss", discriminator_neutral_loss_sum / sample_size / math.log(2),
                    sample_size,
                    round=3
                )

        # for both step 2 & 3

        if "expressive_ncorrect" in logging_outputs[0].keys():
            expressive_ncorrect = [log.get("expressive_ncorrect", 0).item() for log in logging_outputs]
            expressive_incorrect = [log.get("expressive_incorrect", 0).item() for log in logging_outputs]
            # breakpoint()
            expressive_ncorrect_sum = sum(expressive_ncorrect)
            expressive_incorrect_sum = sum(expressive_incorrect)

            neutral_ncorrect = [log.get("neutral_ncorrect", 0).item() for log in logging_outputs]
            neutral_incorrect = [log.get("neutral_incorrect", 0).item() for log in logging_outputs]
            neutral_ncorrect_sum = sum(neutral_ncorrect)
            neutral_incorrect_sum = sum(neutral_incorrect)

            sum_ = sum(expressive_ncorrect + expressive_incorrect + neutral_ncorrect + neutral_incorrect)

            assert sum_ == sample_size, "sample size must equal to sum_"

            metrics.log_scalar(
                "accuracy", (expressive_ncorrect_sum + neutral_ncorrect_sum) / sample_size, sample_size, round=3
            )

            metrics.log_scalar(
                "expressive_accuracy", expressive_ncorrect_sum / sample_size, sample_size, round=3
            )

            n_exp = expressive_ncorrect_sum + expressive_incorrect_sum
            if n_exp != 0:
                exp_norm_acc = expressive_ncorrect_sum / n_exp
            else:
                exp_norm_acc = 0
            metrics.log_scalar(
                "expressive_norm_accuracy", exp_norm_acc, sample_size, round=3
            )

            metrics.log_scalar(
                "neutral_accuracy", neutral_ncorrect_sum / sample_size, sample_size, round=3
            )

            n_neu = neutral_ncorrect_sum + neutral_incorrect_sum
            if n_neu != 0:
                neu_norm_acc = neutral_ncorrect_sum / n_neu
            else:
                neu_norm_acc = 0
            metrics.log_scalar(
                "neutral_norm_accuracy", neu_norm_acc, sample_size, round=3
            )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@dataclass
class RegExtraLabelSmoothedCrossEntropyCriterionConfig(ExtraLabelSmoothedCrossEntropyCriterionConfig):
    lambd: float = field(
        default=0.0,
        metadata={"help": "amount of regularization"}
    )
    schedule_reg: bool = field(
        default=False,
        metadata={"help": "schedule to gradually decrease the reg loss"}
    )
    schedule_reg_rate: float = field(
        default=-0.001,
        metadata={"help": "rate of scheduling"}
    )
    schedule_reg_rate_inc: bool = field(
        default=False,
        metadata={"help": "increase reg rate gradually"}
    )



@register_criterion(
    "reg_extra_label_smoothed_cross_entropy", dataclass=RegExtraLabelSmoothedCrossEntropyCriterionConfig
)
class RegExtraLabelSmoothedCrossEntropyCriterion(ExtraLabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            lambd=0,
            schedule_reg=False,
            schedule_reg_rate=-0.001,
            schedule_reg_rate_inc=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.lambd = lambd
        self.schedule_reg = schedule_reg
        self.schedule_reg_rate = schedule_reg_rate
        self.schedule_reg_rate_inc = schedule_reg_rate_inc

    def forward(self, model, sample, step="st", prev_enc_out=None, proj_speech=False, pass_adapter=False, reduce=True):
        print(f"----  task {step} ------")
        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        #
        decoder_net_output, enc_pool_out = model(**sample["net_input"],
                                                 proj_speech=proj_speech)
        # breakpoint()

        decoder_loss, decoder_nll_loss, decoder_expressive_loss, decoder_neutral_loss = self.compute_decoder_separate_loss(
            model, decoder_net_output, sample, reduce=reduce)

        loss, nll_loss = decoder_loss, decoder_nll_loss
        # self.device = loss.device

        reg_loss = torch.tensor([0], device=loss.device, dtype=loss.dtype)
        if step == "st-reg" and prev_enc_out is not None:
            # breakpoint()
            reg_loss = self.cal_norm2(enc_pool_out, prev_enc_out)
            print("original reg_loss: ", reg_loss)

            reg_loss = self.lambd * reg_loss
            print("reg_loss * lambd: ", reg_loss)
            if self.schedule_reg:
                t = model.encoder.num_updates
                alpha = (1. + np.exp(self.schedule_reg_rate * t)) - 1
                reg_loss = alpha * reg_loss
                print("alpha: ", alpha)
                print("reg_loss * lambd * alpha: ", reg_loss)
            loss += reg_loss


        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "decoder_loss": decoder_loss.data,
            "decoder_nll_loss": decoder_nll_loss.data,
            "reg_loss": reg_loss.data,
            "decoder_expressive_loss": decoder_expressive_loss.data,
            "decoder_neutral_loss": decoder_neutral_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        enc_pool_out = enc_pool_out.detach()

        return loss, sample_size, logging_output, enc_pool_out, decoder_net_output

    def cal_norm2(self, curr_enc_out, prev_enc_out):
        # breakpoint()
        return (prev_enc_out.detach() - curr_enc_out).norm(dim=-1).sum()

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        # for reg method
        # breakpoint()
        ExtraLabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)  # TODO: fix sample_size for exp & neu

        reg_loss_sum = sum(log.get("reg_loss", 0) for log in logging_outputs)
        reg_sample_size = sum(log.get("sample_size", 0) for log in logging_outputs if "reg_loss" in log)
        if reg_sample_size != 0:
            metrics.log_scalar(
                "reg_loss", reg_loss_sum / reg_sample_size / math.log(2), reg_sample_size, round=3
            )

        decoder_loss_sum = sum(log.get("decoder_loss", 0) for log in logging_outputs)
        decoder_nll_loss_sum = sum(log.get("decoder_nll_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "decoder_loss", decoder_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "decoder_nll_loss", decoder_nll_loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        decoder_expressive_loss_sum = sum(log.get("decoder_expressive_loss", 0) for log in logging_outputs)
        decoder_neutral_loss_sum = sum(log.get("decoder_neutral_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "decoder_expressive_loss", decoder_expressive_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "decoder_neutral_loss", decoder_neutral_loss_sum / sample_size / math.log(2), sample_size, round=3
        )


@dataclass
class RegEncoderExtraLabelSmoothedCrossEntropyCriterionConfig(RegExtraLabelSmoothedCrossEntropyCriterionConfig):
    reg_wav2vec_extractor: bool = field(
        default=False,
        metadata={"help": "Whether to regularize the 2 adapters"}
    )
    reg_wav2vec_adapter: bool = field(
        default=False,
        metadata={"help": "Whether to regularize wav2vec post extractor adapter"}
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
    trans_loss_rate: float = field(
        default=1,
        metadata={"help": "coefficient for translation loss"}

    )

    reconstruct_teacher: bool = field(
        default=True,
        metadata={"help": "whether to construct teacher"}
    )
    reg_mod: str = field(
        default="cross_attn_reg",
        metadata={"help": "which regularisation model to use."}
    )


@register_criterion(
    "reg_enc_extra_label_smoothed_cross_entropy", dataclass=RegEncoderExtraLabelSmoothedCrossEntropyCriterionConfig
)
class RegEncoderExtraLabelSmoothedCrossEntropyCriterion(RegExtraLabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            lambd=0,
            schedule_reg=False,
            schedule_reg_rate=-0.001,
            schedule_reg_rate_inc=False,
            trans_loss_rate=1,
            reg_wav2vec_extractor=False,
            reg_wav2vec_adapter=False,
            reg_wav2vec_trans_encoder=False,
            n_trans_encoder_layer=23,
            reg_wav2vec_trans_encoder_reverse=False,
            reg_first_adapter=False,
            reg_len_adapter=False,
            reg_post_len_adapter=False,
            cross_attentive_loss_with_norm=True,
            reconstruct_teacher=True,
            reg_mod="cross_attn_reg"

    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy, lambd, schedule_reg, schedule_reg_rate, schedule_reg_rate_inc)
        self.reg_wav2vec_extractor = reg_wav2vec_extractor
        self.reg_wav2vec_adapter = reg_wav2vec_adapter
        self.reg_wav2vec_trans_encoder = reg_wav2vec_trans_encoder
        self.n_trans_encoder_layer = n_trans_encoder_layer
        self.reg_wav2vec_trans_encoder_reverse = reg_wav2vec_trans_encoder_reverse
        self.trans_loss_rate = trans_loss_rate

        self.reg_first_adapter = reg_first_adapter
        self.reg_len_adapter = reg_len_adapter
        self.reg_post_len_adapter = reg_post_len_adapter

        self.reg_mod = reg_mod
        if self.reg_mod == "cross_attn_reg":
            self.loss_fn = self.cross_attentive_loss
            self.cross_attentive_loss_with_norm = cross_attentive_loss_with_norm
        elif self.reg_mod == "frame_level_reg":
            self.loss_fn = self.frame_level_l2_loss
        elif self.reg_mod == "frame_level_contra":
            self.cos_sim = nn.CosineSimilarity(dim=-1)
            self.ce_loss = nn.CrossEntropyLoss()
            self.loss_fn = self.frame_level_contra_loss
        else:
            raise NotImplementedError

        self.reconstruct_teacher = reconstruct_teacher

    def forward(self, model, sample, step="st", prev_enc_out=None, pass_adapter=False, proj_speech=False, reduce=True):
        # breakpoint()
        split_loss = isinstance(model, Wav2Vec2Seq2SeqModRegEncoderParaOptModel)
        print(
            f"----  task {step} ---")
        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        # breakpoint()
        decoder_net_output, encoder_out = model(**sample["net_input"],
                                                pass_adapter=pass_adapter,
                                                proj_speech=proj_speech)

        decoder_loss, decoder_nll_loss, decoder_expressive_loss, decoder_neutral_loss = self.compute_decoder_separate_loss(
            model, decoder_net_output, sample, reduce=reduce)
        if pass_adapter:
            assert torch.unique(sample['label'].squeeze()).item() == 1, "when passing through the new adapter, all labels have to be 1"
        loss, nll_loss = decoder_loss, decoder_nll_loss
        if self.trans_loss_rate < 1 and model.training:
            loss *= self.trans_loss_rate
        # self.device = loss.device

        reg_loss = torch.tensor([0], device=loss.device, dtype=loss.dtype)
        if model.reg_encoder and step == "st-reg" and prev_enc_out is not None:
            reg_loss = self.cal_reg_loss(encoder_out, prev_enc_out, reg_loss, model, pass_adapter)  # encoder_out as student, prev_enc_out as teacher'

            print("original reg_loss: ", reg_loss)

            reg_loss = self.lambd * reg_loss
            print("reg_loss * lambd: ", reg_loss)
            # breakpoint()
            if self.schedule_reg:
                t = model.encoder.num_updates
                if not self.schedule_reg_rate_inc:
                    alpha = (1. + np.exp(self.schedule_reg_rate * t)) - 1  # -0.001 ~ -0.0001 good number
                else:
                    alpha = 2. / (1. + np.exp(self.schedule_reg_rate * t)) - 1  # -0.001 ~ -0.0001 good number
                reg_loss = alpha * reg_loss
                print("alpha: ", alpha)
                print("reg_loss * lambd * alpha: ", reg_loss)

            if not split_loss:
                loss += reg_loss.squeeze()
            else:
                print("splitting loss to translation loss and reg loss")

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "decoder_loss": decoder_loss.data,
            "decoder_nll_loss": decoder_nll_loss.data,
            "reg_loss": reg_loss.data,
            "decoder_expressive_loss": decoder_expressive_loss.data,
            "decoder_neutral_loss": decoder_neutral_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        # enc_pool_out = encoder_out.detach()
        if split_loss and model.training:
            return (loss, reg_loss), sample_size, logging_output, encoder_out, decoder_net_output
        else:
            return loss, sample_size, logging_output, encoder_out, decoder_net_output

    def cal_reg_loss(self, student_enc_out, teacher_enc_out, reg_loss, model, pass_adapter):

        # print("loss_fn: ", self.loss_fn)

        if self.reg_wav2vec_extractor:
            print("............................  regularizing wav2vec feature extractor ............................")
            assert model.encoder.w2v_model.feature_grad_mult > 0, "feature_grad_mult of wav2vec feature extractor must be greater than 0"
            # breakpoint()
            p0 = next(model.encoder.w2v_model.feature_extractor.parameters())
            assert p0.requires_grad, "feature extractor parameters must require gradient"

            student_extractor_out = student_enc_out["extractor_features"]
            student_extractor_masking = student_enc_out["trans_encoder_padding_mask"]

            teacher_extractor_out = teacher_enc_out["extractor_features"]
            teacher_extractor_masking = teacher_enc_out["trans_encoder_padding_mask"]

            extractor_loss = self.loss_fn(teacher_extractor_out, student_extractor_out,
                                                       teacher_extractor_masking, student_extractor_masking)
            print(f"--- extractor_loss={extractor_loss} ----")
            reg_loss += extractor_loss

        if model.inject_wav2vec_adapter and self.reg_wav2vec_adapter and pass_adapter:
            if model.encoder.wav2vec_version == "wav2vec2_out_adapt":
                print(
                    ".........................  regularizing wav2vec post extractor adapter .......................")
                student_wav2vec_adapter_out = student_enc_out["post_extractor_adapter_features"]
                student_adapter_masking = student_enc_out["trans_encoder_padding_mask"]

                teacher_wav2vec_adapter_out = teacher_enc_out["extractor_features"]
                teacher_adapter_masking = teacher_enc_out["trans_encoder_padding_mask"]

                wav2vec_adapter_loss = self.loss_fn(teacher_wav2vec_adapter_out,
                                                                 student_wav2vec_adapter_out,
                                                                 teacher_adapter_masking, student_adapter_masking)

                reg_loss += wav2vec_adapter_loss
                print(f"--- wav2vec_adapter_loss={wav2vec_adapter_loss} ----")
            elif model.encoder.wav2vec_version == "wav2vec2_in_adapt":
                print(
                    "......................... regularizing wav2vec transformer layer adapter ....................")
                student_wav2vec_adapter_out_list = student_enc_out["trans_encoder_layer_adapter_features"][
                    0]  # output right after the adapter
                student_adapter_masking = student_enc_out["trans_encoder_padding_mask"]

                teacher_wav2vec_adapter_out_list = teacher_enc_out["trans_encoder_layer_adapter_features"][
                    0]  # output right after ffc
                teacher_adapter_masking = teacher_enc_out["trans_encoder_padding_mask"]
                # breakpoint()
                assert len(student_wav2vec_adapter_out_list) == len(
                    teacher_wav2vec_adapter_out_list) == 24, "student and teach must have 24 layers"

                if self.reg_wav2vec_trans_encoder_reverse:
                    start_layer = len(student_wav2vec_adapter_out_list) - 1
                    end_layer = self.n_trans_encoder_layer
                    step_ = -1
                else:
                    start_layer = 0
                    end_layer = self.n_trans_encoder_layer
                    step_ = 1
                for i in range(start_layer, end_layer, step_):
                    print("i-th layer: ", i)
                    student_wav2vec_adapter_out = student_wav2vec_adapter_out_list[i]
                    teacher_wav2vec_adapter_out = teacher_wav2vec_adapter_out_list[i]

                    wav2vec_adapter_loss = self.loss_fn(teacher_wav2vec_adapter_out,
                                                                     student_wav2vec_adapter_out,
                                                                     teacher_adapter_masking, student_adapter_masking)
                    reg_loss += wav2vec_adapter_loss
                    print(f"--- wav2vec_adapter_loss={wav2vec_adapter_loss} ----")
            else:
                NotImplementedError

        if self.reg_wav2vec_trans_encoder:  # when everything but adapter is frozen, it's another way of tuning adapter at a different level of a transformer layer.
            print("............................ regularizing wav2vec transformer encoder ......................")
            # breakpoint()
            student_layer_features_list = student_enc_out["trans_encoder_layer_results"][0]
            student_layer_features_masking = student_enc_out["trans_encoder_padding_mask"]

            teacher_layer_features_list = teacher_enc_out["trans_encoder_layer_results"][0]
            teacher_layer_features_masking = teacher_enc_out["trans_encoder_padding_mask"]

            # breakpoint()
            assert len(student_layer_features_list) == len(
                teacher_layer_features_list) == 24, "student and teach must have 24 layers"

            if self.reg_wav2vec_trans_encoder_reverse:
                start_layer = len(student_layer_features_list) - 1
                end_layer = self.n_trans_encoder_layer
                step_ = -1
            else:
                start_layer = 0
                end_layer = self.n_trans_encoder_layer
                step_ = 1

            for i in range(start_layer, end_layer, step_):
                if i > 23:
                    continue
                print("i-th layer: ", i)
                student_layer_features = student_layer_features_list[i]
                teacher_layer_features = teacher_layer_features_list[i]
                layer_loss = self.loss_fn(teacher_layer_features, student_layer_features,
                                                       teacher_layer_features_masking, student_layer_features_masking)
                print(f"--- layer_loss={layer_loss} ----")
                reg_loss += layer_loss

        if self.reg_first_adapter:
            # adapter
            print("regularizing the first adaptor ..............")
            student_adapter_out = student_enc_out["adapter_out"]
            student_adapter_masking = student_enc_out["trans_encoder_padding_mask"]

            teacher_adapter_out = teacher_enc_out["adapter_out"]
            teacher_adapter_masking = teacher_enc_out["trans_encoder_padding_mask"]

            adapter_loss = self.loss_fn(teacher_adapter_out, student_adapter_out,
                                                     teacher_adapter_masking, student_adapter_masking)
            print(f"--- adapter_loss={adapter_loss} ----")
            reg_loss += adapter_loss

        if self.reg_len_adapter and not self.reg_post_len_adapter:
            # len_adapter

            print("regularizing length adaptor ..............")
            student_len_adapter_out = student_enc_out["encoder_out"]
            student_len_adapter_masking = student_enc_out["encoder_padding_mask"]

            teacher_len_adapter_out = teacher_enc_out["encoder_out"]
            teacher_len_adapter_masking = teacher_enc_out["encoder_padding_mask"]
            # breakpoint()

            len_adapter_loss = self.loss_fn(teacher_len_adapter_out, student_len_adapter_out,
                                                         teacher_len_adapter_masking, student_len_adapter_masking)
            print(f"--- len_adapter_loss={len_adapter_loss} ----")
            reg_loss += len_adapter_loss

        if self.reg_len_adapter and self.reg_post_len_adapter:
            # len_adapter
            print("regularizing length adaptor ..............")
            student_len_adapter_out = student_enc_out["len_adapter_out"]
            student_len_adapter_masking = student_enc_out["encoder_padding_mask"]

            teacher_len_adapter_out = teacher_enc_out["len_adapter_out"]
            teacher_len_adapter_masking = teacher_enc_out["encoder_padding_mask"]

            len_adapter_loss = self.loss_fn(teacher_len_adapter_out, student_len_adapter_out,
                                                         teacher_len_adapter_masking, student_len_adapter_masking)
            print(f"--- len_adapter_loss={len_adapter_loss} ----")
            reg_loss += len_adapter_loss

        if self.reg_post_len_adapter and pass_adapter:
            # post_len_adapter
            print("regularizing post len adaptor ..............")
            student_post_len_adapter_out = student_enc_out["encoder_out"]
            student_post_len_adapter_masking = student_enc_out["encoder_padding_mask"]

            teacher_post_len_adapter_out = teacher_enc_out["encoder_out"]
            teacher_post_len_adapter_masking = teacher_enc_out["encoder_padding_mask"]

            post_len_adapter_loss = self.loss_fn(teacher_post_len_adapter_out, student_post_len_adapter_out,
                                                         teacher_post_len_adapter_masking, student_post_len_adapter_masking)
            print(f"--- post_len_adapter_loss={post_len_adapter_loss} ----")
            reg_loss += post_len_adapter_loss
        # breakpoint()
        return reg_loss

    def frame_level_l2_loss(self, teacher_states, student_states, teacher_masking, student_masking, eps=1e-6):
        # breakpoint()
        if isinstance(teacher_states, list):
            try:
                assert len(teacher_states) == 1, "here, the length must be 1"
                assert len(student_states) == 1, "here, the length must be 1"
            except Exception as e:
                print("e: ", e)
                breakpoint()
            x = teacher_states[0].detach()
            y = student_states[0]
        else:
            x = teacher_states.detach()
            y = student_states
        assert x.requires_grad == False, "Teachers must not have gradient."
        assert x.size() == y.size(), "Teacher and students sizes must be the same; check if adding noise operation is added."
        cost = (x - y).norm(dim=-1)
        return cost.sum()

    def frame_level_contra_loss(self, teacher_states, student_states, teacher_masking, student_masking, eps=1e-6):
        if isinstance(teacher_states, list):
            try:
                assert len(teacher_states) == 1, "here, the length must be 1"
                assert len(student_states) == 1, "here, the length must be 1"
            except Exception as e:
                print("e: ", e)
                breakpoint()
            x = teacher_states[0].detach()
            y = student_states[0]
        else:
            x = teacher_states.detach()
            y = student_states
        assert x.requires_grad == False, "Teachers must not have gradient."
        y = y.transpose(0, 1)
        cos_sim = self.cos_sim(x, y)   # x: [T, 1, D]; y [1, T, D]
        labels = torch.arange(cos_sim.size(0)).long().to(x.device)
        loss = self.ce_loss(cos_sim, labels)
        return loss


    def cross_attentive_loss(
            self, teacher_states, student_states, teacher_masking, student_masking, eps=1e-6
    ):
        # breakpoint()
        if isinstance(teacher_states, list):
            try:
                assert len(teacher_states) == 1, "here, the length must be 1"
                assert len(student_states) == 1, "here, the length must be 1"
            except Exception as e:
                print("e: ", e)
                breakpoint()
            x = teacher_states[0].detach()
            y = student_states[0]
        else:
            x = teacher_states.detach()   # T X B X D
            y = student_states      # T X B X D
        assert x.requires_grad == False, "Teachers must not have gradient."
        # breakpoint()
        x = x.transpose(0, 1)  # from T X B X D to B X T X D
        y = y.transpose(0, 1)
        if self.cross_attentive_loss_with_norm:
            x = x / (x.norm(dim=2, keepdim=True) + eps)  # like layer normalisation?
            y = y / (y.norm(dim=2, keepdim=True) + eps)
        dim = x.size(-1)
        # lengths: batch X seqLen
        sim_scores_xy = torch.bmm(x, y.transpose(1, 2))  # batch X lenx X leny ]
        # sim_scores_xy0 = torch.bmm(x, y.transpose(1, 2))
        if y.dtype == torch.float16:  # ???? why do they convert fp16 to fp32? for better accuracy???
            sim_scores_xy = sim_scores_xy.float()
            y = y.float()
            x = x.float()
        if teacher_masking != []:
            assert len(teacher_masking) == 1
            sim_scores_xy = sim_scores_xy.masked_fill(teacher_masking[0].unsqueeze(-1), float("-inf"))
        if student_masking != []:
            sim_scores_xy = sim_scores_xy.masked_fill(student_masking[0].unsqueeze(1), float("-inf"))
        # do masking
        y_weights = utils.softmax(sim_scores_xy, dim=-1)  # batch X lenx X leny
        if teacher_masking != []:
            y_weights = y_weights.masked_fill(teacher_masking[0].unsqueeze(-1), 0)
        x_reconstruct_from_y = torch.bmm(y_weights, y)  # batch X lenx X dim
        if self.reconstruct_teacher:
            print("reconstruc teacher ....")
            sim_scores_xx = torch.bmm(x, x.transpose(1, 2))  # batch X lenx X lenx ]
            x_weights = utils.softmax(sim_scores_xx, dim=-1)
            if teacher_masking != []:
                x_weights = x_weights.masked_fill(teacher_masking[0].unsqueeze(-1), 0)

            # no gradient for teacher state
            # breakpoint()
            x_reconstruct_from_x = torch.bmm(x_weights, x).detach()
        else:
            print("do not reconstruct teacher ....")
            x_reconstruct_from_x = x.detach()
        cost = (x_reconstruct_from_x - x_reconstruct_from_y).norm(dim=2)  # batch X lenx
        if teacher_masking != []:
            cost = cost.masked_fill(teacher_masking[0], 0)

        if not self.cross_attentive_loss_with_norm:
            cost = cost / dim
        return cost.sum()


@dataclass
class ContrExtraLabelSmoothedCrossEntropyCriterionConfig(ExtraLabelSmoothedCrossEntropyCriterionConfig):
    beta: float = field(
        default=0.0,
        metadata={"help": "amount of contrastive learning term"}
    )


@register_criterion(
    "contr_extra_label_smoothed_cross_entropy", dataclass=ContrExtraLabelSmoothedCrossEntropyCriterionConfig
)
class ContrExtraLabelSmoothedCrossEntropyCriterion(ExtraLabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            beta=0
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.beta = beta
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, model, sample, step="st", prev_enc_pool_out=None, reduce=True):
        print(
            f"----  task {step} ---")
        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        decoder_net_output, enc_pool_out = model(**sample["net_input"])

        decoder_loss, decoder_nll_loss, decoder_expressive_loss, decoder_neutral_loss = self.compute_decoder_separate_loss(
            model, decoder_net_output, sample, reduce=reduce)

        loss, nll_loss = decoder_loss, decoder_nll_loss

        contr_loss = torch.Tensor([0])
        if step == "st-contr" and prev_enc_pool_out is not None:
            contr_loss = self.cal_contr_loss(model, sample, enc_pool_out, prev_enc_pool_out)
            loss += self.beta * contr_loss
            print("contr_loss: ", contr_loss)

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "decoder_loss": decoder_loss.data,
            "decoder_nll_loss": decoder_nll_loss.data,
            "contr_loss": self.beta * contr_loss.data,
            "decoder_expressive_loss": decoder_expressive_loss.data,
            "decoder_neutral_loss": decoder_neutral_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        enc_pool_out = enc_pool_out.detach()
        # breakpoint()
        if model.index_memory_updated and model.training:
            # only updated after contras training is triggered and update exp only; that's why there is a filtering operation
            # breakpoint()
            labels = sample["label"]
            exp_label_idx = (labels == 1).nonzero(as_tuple=True)[0]
            if len(exp_label_idx) != 0:
                mem_idx = sample["mem_index"]
                mem_idx = mem_idx[mem_idx > -1]
                try:
                    assert len(mem_idx) != 0, "mem_idx should not be empty. Something is wrong!"
                except Exception as e:
                    print("error: ", e)
                    breakpoint()

                try:
                    exp_enc_pool_out = enc_pool_out[exp_label_idx, :]
                except Exception as e:
                    print("error: ", e)
                    breakpoint()
                    exp_enc_pool_out = enc_pool_out[exp_label_idx, :]
                print("------- updating feature memory -")
                model.feature_memory.update(exp_enc_pool_out, mem_idx)

        return loss, sample_size, logging_output, enc_pool_out, decoder_net_output

    def cal_contr_loss(self, model, current_sample, curr_enc_out, prev_enc_out):  # curr_enc_out may contain negatives
        # breakpoint()
        z1 = curr_enc_out
        if model.contrastive_bs <= len(
                curr_enc_out):  # no need to retrieve from memory bank to be computationally efficient.
            print("0" * 50)
            z2 = prev_enc_out
        else:
            print("1" * 50)
            extra_negatives = self.retrieve_extra_negatives(model, current_sample, curr_enc_out)
            z2 = torch.cat((prev_enc_out, extra_negatives), 0)

            # ite
        cos_sim = model.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(model.device)

        loss = self.loss_fct(cos_sim, labels)  # cos_sim contains temperatured raw scores which are required for CE
        return loss

    # TODO: may need to rewrite below code to make it more efficient
    def retrieve_extra_negatives(self, model, current_sample, curr_enc_out):
        n_pos = curr_enc_out.size()[0]
        n_item_from_mem = model.contrastive_bs - n_pos  # number of items to be selected from memory banks
        mem_index = current_sample["mem_index"].squeeze(1)
        cand_i_mem_entries = model.index_memory[mem_index]  # candidates to be selected from index memory
        cand_i_mem_entries = cand_i_mem_entries.view(cand_i_mem_entries.size(0) * cand_i_mem_entries.size(1))
        selected_indices = torch.randperm(len(cand_i_mem_entries))[:n_item_from_mem]
        selected_entries = cand_i_mem_entries[selected_indices]
        extra_negatives = model.feature_memory[selected_entries]
        return extra_negatives

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        # for reg method
        # breakpoint()
        ExtraLabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)  # TODO: fix sample_size for exp & neu

        reg_loss_sum = sum(log.get("contr_loss", 0) for log in logging_outputs)
        reg_sample_size = sum(log.get("sample_size", 0) for log in logging_outputs if "contr_loss" in log)
        if reg_sample_size != 0:
            metrics.log_scalar(
                "contr_loss", reg_loss_sum / reg_sample_size / math.log(2), reg_sample_size, round=3
            )

        decoder_expressive_loss_sum = sum(log.get("decoder_expressive_loss", 0) for log in logging_outputs)
        decoder_neutral_loss_sum = sum(log.get("decoder_neutral_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "decoder_expressive_loss", decoder_expressive_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "decoder_neutral_loss", decoder_neutral_loss_sum / sample_size / math.log(2), sample_size, round=3
        )


@register_criterion(
    "dis_extra_label_smoothed_cross_entropy", dataclass=ExtraLabelSmoothedCrossEntropyCriterionConfig
)
class DisExtraLabelSmoothedCrossEntropyCriterion(ExtraLabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # print(sample)
        # breakpoint()

        net_output = model(**sample["net_input"])
        if not self.report_expressive_neutral_loss:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        else:
            loss, nll_loss, expressive_loss, neutral_loss = self.compute_separate_loss(model, net_output, sample,
                                                                                       reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "expressive_loss": expressive_loss.data,
                "neutral_loss": neutral_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }

        if self.report_expressive_neutral_accuracy:
            lprobs, label = self.get_lprobs_and_label(model, net_output, sample)
            expressive_ncorrect, expressive_incorrect, neutral_ncorrect, neutral_incorrect = \
                calculate_expressive_neutral_discriminator_accuracy(lprobs, label)
            logging_output["expressive_ncorrect"] = expressive_ncorrect
            logging_output["neutral_ncorrect"] = neutral_ncorrect
            logging_output["expressive_incorrect"] = expressive_incorrect
            logging_output["neutral_incorrect"] = neutral_incorrect

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, label = self.get_lprobs_and_label(model, net_output,
                                                  sample)  # the last dimention of lprobs is 2 for classification
        # # breakpoint()
        loss, nll_loss = discriminator_label_smoothed_nll_loss(
            lprobs,
            label,
            epsilon=0,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_separate_loss(self, model, net_output, sample, reduce=True):
        lprobs, label = self.get_lprobs_and_label(model, net_output,
                                                  sample)  # the last dimention of lprobs is 2 for classification
        # # breakpoint()
        loss, nll_loss = discriminator_label_smoothed_nll_loss(
            lprobs,
            label,
            epsilon=0,
            reduce=reduce,
        )
        expressive_loss, neutral_loss = calculate_expressive_neutral_discriminator_loss(
            loss,
            lprobs,
            label,
            reduce=True)

        # assert loss.requires_grad == True, nll_loss.requires_grad == True
        assert expressive_loss.requires_grad == False, neutral_loss.requires_grad == False
        return loss, nll_loss, expressive_loss, neutral_loss

    # def get_lprobs_and_label(self, model, net_output, sample):
    #     # # breakpoint()
    #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
    #     # # breakpoint()
    #     _, label = model.get_targets(sample, net_output)
    #     return lprobs, label.view(-1)


@register_criterion(
    "adv_extra_label_smoothed_cross_entropy", dataclass=ExtraLabelSmoothedCrossEntropyCriterionConfig
)
class AdvExtraLabelSmoothedCrossEntropyCriterion(ExtraLabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample, containing translation loss and label loss.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        decoder_net_output, discriminator_net_output = model(**sample["net_input"])
        # breakpoint()
        if not self.report_expressive_neutral_loss:
            decoder_loss, decoder_nll_loss = self.compute_decoder_loss(model, decoder_net_output, sample, reduce=reduce)

            discriminator_loss, discriminator_nll_loss = self.compute_discriminator_loss(model,
                                                                                         discriminator_net_output,
                                                                                         sample, reduce=reduce)

            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )

            loss = decoder_loss + discriminator_loss
            nll_loss = decoder_nll_loss + discriminator_nll_loss
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "decoder_loss": decoder_loss.data,
                "decoder_nll_loss": decoder_nll_loss.data,
                "discriminator_loss": discriminator_loss.data,
                "discriminator_nll_loss": discriminator_nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        else:
            decoder_loss, decoder_nll_loss, decoder_expressive_loss, decoder_neutral_loss = self.compute_decoder_separate_loss(
                model, decoder_net_output, sample, reduce=reduce)
            discriminator_loss, discriminator_nll_loss, discriminator_expressive_loss, discriminator_neutral_loss = self.compute_discriminator_separate_loss(
                model,
                discriminator_net_output,
                sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )

            loss = decoder_loss + discriminator_loss
            nll_loss = decoder_nll_loss + discriminator_nll_loss
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "decoder_loss": decoder_loss.data,
                "decoder_nll_loss": decoder_nll_loss.data,
                "decoder_expressive_loss": decoder_expressive_loss.data,
                "decoder_neutral_loss": decoder_neutral_loss.data,
                "discriminator_loss": discriminator_loss.data,
                "discriminator_nll_loss": discriminator_nll_loss.data,
                "discriminator_expressive_loss": discriminator_expressive_loss.data,
                "discriminator_neutral_loss": discriminator_neutral_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            # print("logging_output: ", logging_output)

        if self.report_expressive_neutral_accuracy:
            # device = loss.device
            lprobs, label = self.get_lprobs_and_label(model, discriminator_net_output, sample)
            expressive_ncorrect, expressive_incorrect, neutral_ncorrect, neutral_incorrect = \
                calculate_expressive_neutral_discriminator_accuracy(lprobs, label)
            logging_output["expressive_ncorrect"] = expressive_ncorrect
            logging_output["neutral_ncorrect"] = neutral_ncorrect
            logging_output["expressive_incorrect"] = expressive_incorrect
            logging_output["neutral_incorrect"] = neutral_incorrect
        # print("loss, sample_size, logging_output: ", loss, sample_size, logging_output)
        return loss, sample_size, logging_output

    # def get_lprobs_and_target(self, model, net_output, sample):
    #     # # breakpoint()
    #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
    #     # # breakpoint()
    #     target, _ = model.get_targets(sample, net_output)
    #     # # breakpoint()
    #     if getattr(lprobs, "batch_first", False):
    #         lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
    #         target = target[:, self.ignore_prefix_size:].contiguous()
    #     else:
    #         lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
    #         target = target[self.ignore_prefix_size:, :].contiguous()
    #     return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def get_lprobs_and_label(self, model, net_output, sample):
        # # breakpoint()
        lprobs = model.get_discriminator_normalized_probs(net_output, log_probs=True)
        # # breakpoint()
        _, label = model.get_targets(sample, net_output)
        return lprobs, label.view(-1)

    # def get_label_only(self, model, net_output, sample):
    #     _, label = model.get_targets(sample, net_output)
    #     return label.view(-1)

    def compute_decoder_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output,
                                                    sample)  # the last dimention of lprobs is either ~250k for translation

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        # breakpoint()
        loss = loss / self.theta
        return loss, nll_loss

    # def compute_decoder_separate_loss(self, model, net_output, sample, reduce=True):
    #     # # breakpoint()
    #     lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
    #     loss, nll_loss = label_smoothed_nll_loss(
    #         lprobs,
    #         target,
    #         self.eps,
    #         ignore_index=self.padding_idx,
    #         reduce=reduce,
    #     )
    #     # breakpoint()
    #     label = self.get_label_only(model, net_output, sample)
    #     expressive_loss, neutral_loss = calculate_expressive_neutral_translation_loss(
    #         loss,
    #         lprobs,
    #         target,
    #         label,
    #         self.eps,
    #         ignore_index=self.padding_idx,
    #         reduce=True)
    #     loss, expressive_loss, neutral_loss = loss / self.theta, \
    #                                           expressive_loss / self.theta, neutral_loss / self.theta
    #     assert expressive_loss.requires_grad == False, neutral_loss.requires_grad == False
    #     return loss, nll_loss, expressive_loss, neutral_loss

    def compute_discriminator_loss(self, model, net_output, sample, reduce=True):
        lprobs, label = self.get_lprobs_and_label(model, net_output,
                                                  sample)  # the last dimention of lprobs 2 for classification
        # # breakpoint()
        loss, nll_loss = discriminator_label_smoothed_nll_loss(
            lprobs,
            label,  # adversarial training
            epsilon=0,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_discriminator_separate_loss(self, model, net_output, sample, reduce=True):
        lprobs, label = self.get_lprobs_and_label(model, net_output, sample)
        loss, nll_loss = discriminator_label_smoothed_nll_loss(
            lprobs,
            label,
            epsilon=0,
            reduce=reduce,
        )
        expressive_loss, neutral_loss = calculate_expressive_neutral_discriminator_loss(
            loss,
            lprobs,
            label,
            reduce=True)
        assert expressive_loss.requires_grad == False, neutral_loss.requires_grad == False
        return loss, nll_loss, expressive_loss, neutral_loss


@register_criterion(
    "mul_adv_extra_label_smoothed_cross_entropy", dataclass=ExtraLabelSmoothedCrossEntropyCriterionConfig
)
class MulAdvExtraLabelSmoothedCrossEntropyCriterion(AdvExtraLabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)

    def forward(self, model, sample, step="translator", reduce=True):
        """Compute the loss for the given sample, containing translation loss and label loss.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # # breakpoint()
        # print("9"*100)

        decoder_net_output, discriminator_net_output = model(**sample["net_input"])
        # print("sample[target].size(0): ", sample["target"].size(0))
        # print("sample[ntokens]: ", sample["ntokens"])
        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        # print("sample_size: ", sample_size)
        # breakpoint()
        if not self.report_expressive_neutral_loss:
            if step == "translator":
                print(f"----- training for task {step} ----")

                decoder_net_output, _ = model(**sample["net_input"])
                decoder_loss, decoder_nll_loss = self.compute_decoder_loss(model, decoder_net_output, sample,
                                                                           reduce=reduce)

                loss, nll_loss = decoder_loss, decoder_nll_loss
                device = loss.device
                discriminator_loss, discriminator_nll_loss = \
                    torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)
            elif step == "discriminator":
                print(f"----- training for task {step} ----")
                _, discriminator_net_output = model(**sample["net_input"])
                discriminator_loss, discriminator_nll_loss = self.compute_discriminator_loss(model,
                                                                                             discriminator_net_output,
                                                                                             sample, reduce=reduce)
                loss, nll_loss = discriminator_loss, discriminator_nll_loss
                device = loss.device
                decoder_loss, decoder_nll_loss = torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)
            elif step == "both":
                print(f"----- evaluation for task {step} ----")
                # below for valuation
                decoder_net_output, discriminator_net_output = model(**sample["net_input"])
                decoder_loss, decoder_nll_loss = self.compute_decoder_loss(model, decoder_net_output, sample,
                                                                           reduce=reduce)
                discriminator_loss, discriminator_nll_loss = self.compute_discriminator_loss(model,
                                                                                             discriminator_net_output,
                                                                                             sample, reduce=reduce)
                loss = decoder_loss + discriminator_loss
                nll_loss = decoder_nll_loss + discriminator_nll_loss
            else:
                raise NotImplementedError

            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "decoder_loss": decoder_loss.data,
                "decoder_nll_loss": decoder_nll_loss.data,
                "discriminator_loss": discriminator_loss.data,
                "discriminator_nll_loss": discriminator_nll_loss.data,
                # "discrim_step": discrim_step,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        else:
            if step == "translator":
                print(f"----- training for task {step} ----")
                decoder_net_output, _ = model(**sample["net_input"])
                decoder_loss, decoder_nll_loss, decoder_expressive_loss, decoder_neutral_loss = self.compute_decoder_separate_loss(
                    model, decoder_net_output, sample, reduce=reduce)

                loss, nll_loss = decoder_loss, decoder_nll_loss
                device = loss.device
                discriminator_loss, discriminator_nll_loss, discriminator_expressive_loss, discriminator_neutral_loss = \
                    torch.tensor([0.0], device=device), torch.tensor([0.0], device=device), torch.tensor([0.0],
                                                                                                         device=device), torch.tensor(
                        [0.0], device=device)
            elif step == "discriminator":
                print(f"----- training for task {step} ----")
                _, discriminator_net_output = model(**sample["net_input"])
                discriminator_loss, discriminator_nll_loss, discriminator_expressive_loss, discriminator_neutral_loss = self.compute_discriminator_separate_loss(
                    model,
                    discriminator_net_output,
                    sample, reduce=reduce)
                loss, nll_loss = discriminator_loss, discriminator_nll_loss
                device = loss.device
                decoder_loss, decoder_nll_loss, decoder_expressive_loss, decoder_neutral_loss = \
                    torch.tensor([0.0], device=device), torch.tensor([0.0], device=device), torch.tensor([0.0],
                                                                                                         device=device), torch.tensor(
                        [0.0], device=device)
            elif step == "both":
                # below for valuation
                print(f"----- evaluation for task {step} ----")
                decoder_net_output, discriminator_net_output = model(**sample["net_input"])
                decoder_loss, decoder_nll_loss, decoder_expressive_loss, decoder_neutral_loss = self.compute_decoder_separate_loss(
                    model, decoder_net_output, sample, reduce=reduce)
                discriminator_loss, discriminator_nll_loss, discriminator_expressive_loss, discriminator_neutral_loss = self.compute_discriminator_separate_loss(
                    model,
                    discriminator_net_output,
                    sample, reduce=reduce)
                loss = decoder_loss + discriminator_loss
                nll_loss = decoder_nll_loss + discriminator_nll_loss
            else:
                raise NotImplementedError

            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "decoder_loss": decoder_loss.data,
                "decoder_nll_loss": decoder_nll_loss.data,
                "decoder_expressive_loss": decoder_expressive_loss.data,
                "decoder_neutral_loss": decoder_neutral_loss.data,
                "discriminator_loss": discriminator_loss.data,
                "discriminator_nll_loss": discriminator_nll_loss.data,
                "discriminator_expressive_loss": discriminator_expressive_loss.data,
                "discriminator_neutral_loss": discriminator_neutral_loss.data,
                # "discrim_step": discrim_step,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        # breakpoint()
        if (step == "discriminator" or step == "both") and self.report_expressive_neutral_accuracy:
            lprobs, label = self.get_lprobs_and_label(model, discriminator_net_output, sample)
            expressive_ncorrect, expressive_incorrect, neutral_ncorrect, neutral_incorrect = \
                calculate_expressive_neutral_discriminator_accuracy(lprobs, label)
            logging_output["expressive_ncorrect"] = expressive_ncorrect
            logging_output["neutral_ncorrect"] = neutral_ncorrect
            logging_output["expressive_incorrect"] = expressive_incorrect
            logging_output["neutral_incorrect"] = neutral_incorrect
        return loss, sample_size, logging_output
