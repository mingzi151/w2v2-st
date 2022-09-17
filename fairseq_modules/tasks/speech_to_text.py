import numpy as np
from omegaconf import II
from dataclasses import dataclass, field
import os.path as op

from fairseq.data import ConcatDataset, SubsampleDataset
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import (
    SpeechToTextTask,
    SpeechToTextTaskConfig,
)
import json
import random
random.seed(5)
from fairseq.distributed import utils as distributed_utils
import sacrebleu
from fairseq_modules.data.control_subsample_dataset import ControlSubsampleDataset

from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform
)
from fairseq.logging import meters, metrics, progress_bar
import itertools as it
from collections import Counter
import math
import os, time
from fairseq.data.dictionary import LabelDictionary
import torch
from fairseq import metrics, search, tokenizer, utils
from fairseq.data import Dictionary, FairseqDataset, data_utils, encoders, iterators

from fairseq_modules.data.augmentation_normalization_dataset import AugmentationNormalizationDataset
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

from fairseq_modules.models.wav2vec_s2t import Wav2Vec2Seq2SeqModRegEncoderParaOptModel
import logging

import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
import numpy as np


# We'll hack a bit with the t-SNE code in sklearn.
from sklearn.metrics.pairwise import pairwise_distances

# from sklearn.utils.extmath import _ravel
# Random state.
RS = 25111993

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

logger = logging.getLogger(__name__)


def warm_up(model, sample, repeat=10):
    """Warm up the model"""
    print("warming up ...")
    model.eval()
    # breakpoint()
    for _i in range(repeat):
        model.forward(**sample["net_input"])
    logger.info(f"Model warmed up by running inference {repeat} times")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def throughput(model, sample, repeat=30):
    model.eval()
    # for idx, (images, _) in enumerate(data_loader):
    if not sample["net_input"]['src_tokens'].is_cuda:
        sample["net_input"]['src_tokens'] = sample["net_input"]['src_tokens'].cuda(non_blocking=True)
    # sample = sample.cuda(non_blocking=True)
    batch_size = sample["net_input"]['src_tokens'].shape[0]

    print("#"*50)
    print("batch_size: ", batch_size)

    warm_up(model, sample)

    print()
    print(f"throughput averaged with {repeat} times for encoder+decoder")
    torch.cuda.synchronize()
    tic1 = time.time()
    for i in range(repeat):
        model(**sample["net_input"])
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"==> batch_size {batch_size} throughput {repeat * batch_size / (tic2 - tic1)}")

    print()
    print(f"throughput averaged with {repeat} times for encoder only")
    torch.cuda.synchronize()
    tic3 = time.time()
    for i in range(repeat):
        model.encoder(**sample["net_input"])
    torch.cuda.synchronize()
    tic4 = time.time()
    print(f"==> batch_size {batch_size} throughput {repeat * batch_size / (tic4 - tic3)}")
    # breakpoint()
    return

# macs, params = get_model_complexity_info(model, (sample["net_input"],), as_strings=True, input_constructor=get_first_item, print_per_layer_stat=True, verbose=True)

def get_lnorm_flops(module, input, output):
    # breakpoint()
    # print("!!!!!!!!!!!!! module: ", module)
    input = input[0]
    B, N, C = input.size()[0], input.size()[1], input.size()[2]
    flops = 0
    # in_features
    in_features, out_features = C, output.size()[2]
    assert in_features == module.normalized_shape[0], f"input feature dim must equal to {module} dim."
    flops += B * N * in_features * out_features
    module.__flops__ += flops

# dir(model.w2v_model.feature_extractor.conv_layers[1][2][1])

# macs, params = get_model_complexity_info(model, (sample["net_input"],), as_strings=True,input_constructor=get_first_item,print_per_layer_stat=False, verbose=True, custom_modules_hooks={Fp32LayerNorm: get_lnorm_flops, LayerNorm: get_lnorm_flops})

def get_msa_flops(module, input, extra):
    # TODO: add dropout for training
#    breakpoint()
    extra = extra[0]
    B, N, C = extra.size()[1], extra.size()[0], extra.size()[2]
    flops = 0
    # x -> q, k, v
    flops += N * module.embed_dim * 3 * module.embed_dim
    # q @ k
    flops += B * module.num_heads * N * (module.embed_dim // module.num_heads) * N
    # attn @ v
    flops += B * module.num_heads * N * N * (module.embed_dim // module.num_heads)
    # proj(x)
    flops += B * N * module.embed_dim * module.embed_dim
    module.__flops__ += flops


def count_flops(
        model,
        sample,
        repeat=20,
):
    """Use PYPAPI library to count average flops for model inference.
    Note: It only works if the model is being run on cpu"""
    from ptflops import get_model_complexity_info

    def get_first_item(tuple_):
        return tuple_[0]

    # breakpoint()
    # macs, params = get_model_complexity_info(model, (sample["net_input"],), as_strings=True,input_constructor=get_first_item,print_per_layer_stat=True, verbose=True, custom_modules_hooks={Fp32LayerNorm: get_lnorm_flops, LayerNorm: get_lnorm_flops})
    macs, params = get_model_complexity_info(model, (sample["net_input"],), as_strings=True, input_constructor=get_first_item,
                                             print_per_layer_stat=True, verbose=True, custom_modules_hooks={
            Fp32LayerNorm: get_lnorm_flops,
            LayerNorm: get_lnorm_flops,
            MultiheadAttention: get_msa_flops,
        })
    # macs, params = get_model_complexity_info(model, (sample["net_input"],), as_strings=True,
    #                                          input_constructor="identity",
    #                                          print_per_layer_stat=True, verbose=True, custom_modules_hooks={
    #         # Fp32LayerNorm: get_lnorm_flops,
    #         # LayerNorm: get_lnorm_flops,
    #         MultiheadAttention: get_msa_flops,
    #     })
    # macs, params = get_model_complexity_info(model, sample["net_input"], as_strings=True,print_per_layer_stat=False, verbose=False)
    print("calculating flops ...")
    # breakpoint()

    # for _r in range(repeat):
    #     macs, params = profile(model, inputs=((**sample["net_input"]),))
    # macs0, params0 = profile(model, inputs=(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"]))
    #
    # macs, params = profile(model, inputs=(tokens, lengths))

        # model.forward(**sample["net_input"])
    print("macs: ", macs)
    print("params: ", params)

    print("exiting program...")
    exit(0)
    # return flops
# def max_memory(model, sample, repeat=30):
#     """Compute average max memory consumed by model inference. Units are MiB"""
#     logger.info("Starting memory benchmarking")
#     from memory_profiler import memory_usage
#     total_memory = 0
#     print("calculating memory ...")
#     # breakpoint()
#     # for i, sample in enumerate(dataset):
#
#     for _r in range(repeat):
#         total_memory += max(memory_usage((model.forward, (sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"]), {})))
#         # if i % 100 == 0:
#         #     logger.info(f"Benchmarked memory for {i}/{len(dataset)} samples")
#     total_memory = total_memory / (repeat * len(sample))
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     return total_memory

def cal_model_mem_infer(model, sample):
    model(**sample["net_input"])
    return

def cal_model_mem_train(model, repeat=2):
    print("cal_model_mem_train  ....")
    pass

def get_batch_size(model, sample):
    model.eval()

    if not sample["net_input"]['src_tokens'].is_cuda:
        sample["net_input"]['src_tokens'] = sample["net_input"]['src_tokens'].cuda(non_blocking=True)
    # sample = sample.cuda(non_blocking=True)
    batch_size = sample["net_input"]['src_tokens'].shape[0]
    print("#"*50)
    print("batch_size: ", batch_size)
    with open("batch_size.txt", "a") as f:
        f.write(str(batch_size)+"\n")
    return

@dataclass
class SpeechToTextModTaskConfig(SpeechToTextTaskConfig):
    sample_ratios: str = field(
        default="1",
        metadata={"help": "sample ratios of the train subsets"}
    )

    da_p_augm: float = field(
        default="1",
        metadata={"help": "The probability that data augmentation is applied to an example."}
    )

    da_tempo: str = field(
        default="1,1",
        metadata={"help": "The range from which to sample the tempo factor during data augmentation"}
    )

    da_pitch: str = field(
        default="0,0",
        metadata={"help": "The range from which to sample the pitch value during data augmentation. \
            Measured in cents (i.e. 100ths of a semitone)"}
    )

    da_echo_delay: str = field(
        default="0,0",
        metadata={"help": "The range from which to sample the echo delay value during data augmentation. \
            Measured in milliseconds"}
    )

    da_echo_decay: str = field(
        default="0,0",
        metadata={"help": "The range from which to sample the echo decay factor during data augmentation."}
    )

    normalize: bool = field(
        default=True,
        metadata={"help": "Whether to normalize the audiowave to zero mean and unit variance."}
    )

    report_expressive_neutral_loss: bool = field(
        default=True,
        metadata={"help": "Whether to report expressive and neutral loss"}
    )

    report_expressive_neutral_accuracy: bool = field(
        default=True,
        metadata={"help": "Whether to report expressive and neutral accuracy"}
    )
    theta: float = field(
        default=1.0,
        metadata={"help": "reduce translation decoder's loss scale"}
    )
    make_even_batch: bool = field(
        default=False,
        metadata={"help": "Whether to merge two singlular batches. "}
    )
    same_samples: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the same datasets at each iteration. Meanwhile, it keeps choosen samples incremental in our experiments "}
    )
    lamb: float = field(
        default=0.0,
        metadata={"help": "amount of regularization"}
    )
    regularize: bool = field(
        default=True,
        metadata={"help": "for regularised st"}
    )
    throughput: bool = field(
        default=False,
        metadata={"help": "calculate throughput only"}
    )
    get_batch_size: bool = field(
        default=False,
        metadata={"help": "get all batch sizes"}
    )
    cal_model_mem_infer: bool = field(
        default=False,
        metadata={"help": "get mem for encoder at inference time"}
    )
    cal_st_mem_infer: bool = field(
        default=False,
        metadata={"help": "get mem for encoder + decoder at inference time"}
    )
    cal_model_mem_train: bool = field(
        default=False,
        metadata={"help": "get FLOPs and mem at training time"}
    )

    cal_flops: bool = field(
        default=False,
        metadata={"help": "get FLOPs at inference time"}
    )
    frame_size: int = field(
        default=5000,
        metadata={"help": "the average number of frames at inference time"}
    )
    #
    # num_pseudo_spk: int = field(
    #     default=0,
    #     metadata={"help": "the average number of frames at inference time"}
    # )

    seed: int = II("common.seed")
    max_tokens: int = II("dataset.max_tokens")


@register_task("speech_to_text_iwslt21", dataclass=SpeechToTextModTaskConfig)
class SpeechToTextModTask(SpeechToTextTask):

    def __init__(self, cfg, tgt_dict, label_dict):
        super().__init__(cfg, tgt_dict)

        self.label_dict = label_dict
        # self.src_dict = src_dict
        self.theta = cfg.theta
        self.make_even_batch = cfg.make_even_batch
        self.same_samples = cfg.same_samples
        # self.num_pseudo_spk = cfg.num_pseudo_spk

        # effect parameters for data augmentation
        self.da_p_augm = cfg.da_p_augm
        self.da_effects_info = {
            "tempo": list(map(float, cfg.da_tempo.split(","))),
            "pitch": list(map(int, cfg.da_pitch.split(","))),
            "echo": {
                "delay": list(map(int, cfg.da_echo_delay.split(","))),
                "decay": list(map(float, cfg.da_echo_decay.split(",")))
            }
        }
        self.report_expressive_neutral_loss = cfg.report_expressive_neutral_loss
        self.report_expressive_neutral_accuracy = cfg.report_expressive_neutral_accuracy

        self.max_src_len = min(cfg.max_source_positions, cfg.max_tokens)
        self.THROUGHPUT_MODE = cfg.throughput
        self.get_batch_size = cfg.get_batch_size
        self.cal_model_mem_infer = cfg.cal_model_mem_infer
        self.cal_st_mem_infer = cfg.cal_st_mem_infer
        self.cal_model_mem_train = cfg.cal_model_mem_train
        self.cal_flops = cfg.cal_flops
        self.frame_size = cfg.frame_size

        assert len(set(self.da_effects_info["echo"]["delay"])) == \
               len(set(self.da_effects_info["echo"]["decay"])), \
            "Specify ranges for both parameters of echo (delay & decay) or for none"
        # breakpoint()

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        data_cfg = S2TDataConfig(op.join(cfg.data, cfg.data_config_yaml))
        dict_path = op.join(cfg.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        label_dict = LabelDictionary()

        logger.info(
            f"target dictionary size ({data_cfg.vocab_filename}): {len(tgt_dict)}"
        )
        logger.info(
            f"label dictionary size: {len(label_dict)}"
        )

        if getattr(cfg, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in cfg.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(cfg, tgt_dict, label_dict)

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if self.THROUGHPUT_MODE:
            # print("...............calculating throughout......... ")
            throughput(models[0], sample)
            # models[0].forward_throughput(**sample["net_input"])
        elif self.get_batch_size:
            get_batch_size(models[0], sample)
        elif self.cal_model_mem_infer:
            # print("...............calculating model stat......... ")
            # with torch.no_grad():
            # del sample
            cal_model_mem_infer(models[0].encoder, sample)
        elif self.cal_model_mem_train:
            # print("...............calculating model stat......... ")
            cal_model_mem_train(models[0].encoder)
        elif self.cal_flops:
            count_flops(models[0].encoder, sample)
        else:
            with torch.no_grad():
                return generator.generate(
                    models, sample, prefix_tokens=prefix_tokens, constraints=constraints
                )



    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        # print("00000000000000 split: ", split)
        # print("4" * 100)
        # breakpoint()
        is_train_split = split.startswith("train")
        if is_train_split:
            datasets = []
            splits = split.split(',')
            sample_ratios = \
                [float(r) for r in self.cfg.sample_ratios.split(',')]
            if sample_ratios == [1]:
                sample_ratios = sample_ratios * len(splits)
            assert len(splits) == len(sample_ratios), \
                "The nº of sfplits and sample_ratios must be equal."
            # breakpoint()
            for s, r in zip(splits, sample_ratios):
                super().load_dataset(s, epoch, combine, **kwargs)
                if 0 < r < 1:
                    print("...  sampling data  ...")
                    # assert self.same_samples == True
                    datasets.append(ControlSubsampleDataset(self.datasets.pop(s), r, same_samples=self.same_samples))
                else:
                    datasets.append(self.datasets.pop(s))
            upsample_ratios = [int(r) if r > 1 else 1 for r in sample_ratios]
            self.datasets[split] = ConcatDataset(datasets, upsample_ratios)
        else:
            # breakpoint()
            super().load_dataset(split, epoch, combine, **kwargs)

        # print(self.da_effects_info, self.da_p_augm)
        # breakpoint()
        self.datasets[split] = AugmentationNormalizationDataset(
            self.datasets[split], self.da_effects_info,
            self.da_p_augm, self.cfg.normalize,
            self.max_src_len, is_train_split)

    def get_batch_iterator(
            self,
            dataset,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
            num_workers=0,
            epoch=1,
            data_buffer_size=0,
            disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        # breakpoint()
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            # breakpoint()
            indices = dataset.ordered_indices()
        print("indices: ", indices)
        # breakpoint()
        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(indices, dataset, max_positions, ignore_invalid_inputs)

        def batchify_by_batch_size(indices, bs):
            # breakpoint()
            assert (len(indices) % 2) == 0, "the number of data must be even"
            assert (bs % 2) == 0, "batch size must be even"
            batchified_indices = [indices[x:x + bs] for x in range(0, len(indices), bs)]
            return batchified_indices

        # breakpoint()
        # create mini-batches with given size constraints
        if not self.make_even_batch:
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = batchify_by_batch_size(indices, bs=max_sentences)
        # breakpoint()
        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def begin_epoch(self, epoch, model):
        super().begin_epoch(epoch, model)
        np.random.seed(self.cfg.seed + epoch)

        if epoch == 1:
            return
        sample_ratios = \
            [float(r) for r in self.cfg.sample_ratios.split(',')]
        # breakpoint()
        if len(set(sample_ratios)) == 1 or self.cfg.same_samples:
            return
        # below code is only needed if different sampling rates are used
        for split in self.datasets.keys():
            print("split: ", split)
            if split.startswith("train"):
                # Perform a new subsampling at each epoch
                self.load_dataset(split, epoch)

    def optimizer_step(self, optimizer, model, update_num):
        # breakpoint()
        if hasattr(model, "get_groups_for_update"):
            print("--------------   model has get_groups_for_update --------------")
            groups = model.get_groups_for_update(update_num)
            print("groups=", groups)
            # trans_opt_params = list(optimizer.optimizers["translator"].params)  # parameters that will be updated
            # dis_opt_params = list(optimizer.optimizers["discriminator"].params)
            if isinstance(groups, tuple):
                assert len(groups) == 2, "There must be two groups"
                optimizer.step(groups={groups[0]})
                optimizer.step(groups={groups[1]})
            else:
                optimizer.step(groups={groups})
        else:
            # print("--------------   model doesn't have get_groups_for_update --------------")
            optimizer.step()


@dataclass
class SpeechToTextModInfTaskConfig(SpeechToTextModTaskConfig):
    perturb_ratio: float = field(
        default=0.2,
        metadata={"help": "sample ratios of the train subsets"}
    )
    perturb_pos: str = field(
        default="input",
        metadata={"help": "sample ratios of the train subsets"}
    )
    get_spch_repre: bool = field(
        default=False,
        metadata={"help": "whether to produce a single vector to represent speech at inference"}
    )
    spch_repre_fname: str = field(
        default="",
        metadata={"help": "json file name for storing spch representations (include avg and first)"}
    )
    skip_global: bool = field(
        default=False,
        metadata={"help": "whether to skip MPSA"}
    )


@register_task("speech_to_text_iwslt21_infer", dataclass=SpeechToTextModInfTaskConfig)
class SpeechToTextModInfTask(SpeechToTextModTask):

    def __init__(self, cfg, tgt_dict, label_dict):
        # breakpoint()
        super().__init__(cfg, tgt_dict, label_dict)
        self.perturb_ratio = cfg.perturb_ratio
        self.perturb_pos = cfg.perturb_pos
        self.spch_repre_fname = cfg.spch_repre_fname
        self.skip_global = cfg.skip_global

    def build_model(self, cfg):
        from fairseq import models, quantization_utils
        model = models.build_model(cfg, self)
        # breakpoint()
        if self.perturb_ratio > 0:
            model.encoder.perturb_ratio = self.perturb_ratio
            model.encoder.perturb_pos = self.perturb_pos
        if self.spch_repre_fname:
            os.remove(self.spch_repre_fname) if os.path.exists(self.spch_repre_fname) else None
            model.encoder.spch_repre_fname = self.spch_repre_fname
        if self.skip_global:
            model.encoder.skip_global = self.skip_global
        model = quantization_utils.quantize_model_scalar(model, cfg)
        return model

@dataclass
class SpeechToTextRegTaskConfig(SpeechToTextModTaskConfig):
    order_frames: bool = field(
        default=False,
        metadata={"help": "sample ratios of the train subsets"}
    )
    skip_even_update: bool = field(
        default=False,
        metadata={"help": "whether skip updating of neutral speech at even update. "}
    )
    warmup_neutral_step: int = field(
        default=-1,
        metadata={"help": "whether skip updating of neutral speech at even update. "}
    )
    skip_update_paired_neu: bool = field(
        default=False,
        metadata={
            "help": "skip updating parameters for paired neutral speech. Instead, update them with unpaired data."}
    )
    reduce_neutral_ce_loss: bool = field(
        default=False,
        metadata={"help": "reduce cross entropy loss term for neutral speech."}
    )
    reduce_neutral_ce_loss_rate: float = field(
        default=1,
        metadata={"Help": "degree of reducing cross entropy loss term for neutral speech."}
    )
    split_loss: bool = field(
        default=False,
        metadata={"help": "Whether to return more than one loss to tune different parts of the model."}
    )


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # print("plotting")
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            try:
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
            except Exception as e:
                print("error: ", e)
                breakpoint()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    breakpoint()


def retrieve_neu_exp_indices(sample):
    label = sample["label"]
    neu_indices = (label == 0).nonzero(as_tuple=True)[0]
    exp_indices = (label == 1).nonzero(as_tuple=True)[0]

    return neu_indices, exp_indices


def check_paired_neu_exp(sample):
    neu_indices, exp_indices = retrieve_neu_exp_indices(sample)
    tgt = sample["target"]
    uni_tgt = torch.unique(tgt, dim=0)
    try:
        return len(tgt) == (len(uni_tgt) * 2) and (neu_indices.size() == exp_indices.size())
    except Exception as e:
        print("error:", e)


def split_by_indices(sample, fir_indices, sec_indices):
    samples = []
    val_key = ['id', 'target', 'target_lengths', 'label', 'mem_index']
    net_input = sample["net_input"]
    net_input_val_key = ['src_lengths', 'prev_output_tokens', 'src_tokens']
    for indices in [fir_indices, sec_indices]:  # always compute neu first !
        if len(indices) == 0:
            samples.append({})
            continue
        temp = {key: sample[key][exp] for sample, key, exp in zip(it.repeat(sample), val_key, it.repeat(indices))}
        temp["nsentences"] = len(indices)
        temp["ntokens"] = sum(temp["target_lengths"])

        temp["net_input"] = {key: net_input[key][exp] for net_input, key, exp in
                             zip(it.repeat(net_input), net_input_val_key, it.repeat(indices))}
        max_src_leng = max(temp["net_input"]["src_lengths"])  # update padding in current examples.
        temp["net_input"]["src_tokens"] = temp["net_input"]["src_tokens"][:, :max_src_leng]
        assert temp.keys() == sample.keys(), "Make sure the split dic has the same keys as sample"

        samples.append(temp)
    return samples


def split_to_half(sample):
    bs = sample["nsentences"]
    cutoff = math.ceil(bs / 2)
    fir_indices = torch.tensor([i for i in range(cutoff)])
    sec_indices = torch.tensor([i for i in range(cutoff, bs)])
    return split_by_indices(sample, fir_indices, sec_indices)


def split_to_neu_and_exp(sample):
    neu_indices, exp_indices = retrieve_neu_exp_indices(sample)
    # try:
    #     assert torch.all((neu_indices + 1).eq(exp_indices)), "The neu and exp indices must match!!, as this is how the data was prepared. "
    # except Exception as e:
    #     print("error: ", e)
    #     breakpoint()
    try:
        return split_by_indices(sample, neu_indices, exp_indices)
    except Exception as e:
        print("error: ", e)
        breakpoint()
        return split_by_indices(sample, neu_indices, exp_indices)

# def check_neu_exp_indices(sample):
#     neu_indices, exp_indices = retrieve_neu_exp_indices(sample)
#     return torch.all((neu_indices + 1).eq(exp_indices))
#

from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


@register_task("speech_to_text_reg", dataclass=SpeechToTextRegTaskConfig)
class SpeechToTextRegTask(SpeechToTextModTask):

    def __init__(self, cfg, tgt_dict, label_dict):
        super().__init__(cfg, tgt_dict, label_dict)
        # self.lamb = cfg.lamb
        # breakpoint()
        self.regularize = cfg.regularize
        self.order_frames = cfg.order_frames
        self.reduce_neutral_ce_loss = cfg.reduce_neutral_ce_loss
        self.reduce_neutral_ce_loss_rate = cfg.reduce_neutral_ce_loss_rate

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        if check_paired_neu_exp(
                sample) and self.regularize:  # set regularize to be False as a sanity check for my code that only st is performed.
            print("--------- splitting samples to neu and exp --------")
            #
            samples = split_to_neu_and_exp(sample)  # always compute neu first !
            # breakpoint()
           # assert samples[0]['id'] + 1 == samples[1]['id'], "the first items must be neu while the second exp."
            steps = ["st", "st-reg"]
        else:
            # breakpoint()
            # samples = [sample, None]
            samples = split_to_half(sample)  # len(samples) = 2
            steps = ["st", "st"]
        return self._train_step(samples, model, criterion, optimizer, update_num, steps, ignore_grad=ignore_grad)

    def _train_step(self, samples, model, criterion, optimizer, update_num, steps=None, ignore_grad=False):
        if steps is None:
            steps = ["st", "st"]
        total_loss, total_sample_size, total_logging_output = 0, 0, Counter()
        sample_size = None
        enc_out = None

        optimizer.zero_grad()
        model.zero_grad()
        # breakpoint()
        for i, (step, sample) in enumerate(zip(steps, samples)):
            if not sample:
                continue
            print("=" * 100)
            print("sample id & label for this update: ", sample["id"], sample["label"])
            print("step to be completed: ", step)
            # prev_sample_size = sample_size

            model.train()
            # breakpoint()
            model.set_num_updates(update_num)
            proj_speech = True if steps == ["st", "st-reg"] and step == "st-reg" else False
            pass_adapter = True if steps == ["st",
                                             "st-reg"] and step == "st-reg" and \
                                   (model.inject_wav2vec_adapter or model.inject_post_len_adapter) else False
            print("pass_adapter: ", pass_adapter)
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output, enc_out, decoder_net_output = criterion(model, sample, step=step,
                                                                                           prev_enc_out=enc_out,
                                                                                           pass_adapter=pass_adapter,
                                                                                           proj_speech=proj_speech)
            # breakpoint()
            if not isinstance(loss, tuple):  # a single optimizer
                if self.reduce_neutral_ce_loss and steps == ["st", "st-reg"] and step == "st":
                    loss *= self.reduce_neutral_ce_loss_rate
                # breakpoint()
                if ignore_grad:
                    loss *= 0
                print("total loss at the train step: ", loss)

                with torch.autograd.profiler.record_function("backward"):
                    if loss and loss.requires_grad:
                        print("generating gradients...... ")
                        try:
                            optimizer.backward(loss)
                        except Exception as e:
                            print("error: ", e)
                            breakpoint()

                # plot_grad_flow(model.named_parameters())
                # breakpoint()
                total_loss += float(loss)

                del loss
            else:  # 2 parallel optmizers
                trans_loss, reg_loss = loss
                if self.reduce_neutral_ce_loss and steps == ["st", "st-reg"] and step == "st":
                    trans_loss *= self.reduce_neutral_ce_loss_rate
                # breakpoint()
                if ignore_grad:
                    trans_loss *= 0
                print("trans_loss at the train step: ", trans_loss)
                print("reg_loss at the train step: ", reg_loss)
                with torch.autograd.profiler.record_function("backward"):
                    groups = model.get_groups_for_update(update_num)

                    if trans_loss and not reg_loss:
                        # breakpoint()
                        # print("0" * 50)
                        print("generating gradients...... ")
                        optimizer.optimizers[groups[0]].backward(trans_loss)
                    if not trans_loss and reg_loss:
                        # print("1" * 50)
                        print("generating gradients...... ")
                        optimizer.optimizers[groups[1]].backward(reg_loss)
                    if trans_loss and reg_loss:
                        # print("2" * 50)
                        # breakpoint()
                        optimizer.optimizers[groups[0]].backward(trans_loss, retain_graph=True)
                        optimizer.optimizers[groups[1]].backward(reg_loss)

                total_loss += float(trans_loss) + float(reg_loss)
                del trans_loss, reg_loss

            total_sample_size += sample_size
            total_logging_output = total_logging_output + Counter(logging_output)

            if torch.cuda.is_available() and update_num == 0:
                torch.cuda.empty_cache()
        # m_para = list(model.parameters())

        # print("m_para: ", m_para[-2:])
        # breakpoint()
        print("total loss at this whole train step:", total_loss)
        return total_loss, total_sample_size, dict(total_logging_output)

    def valid_step(self, sample, model, criterion):
        model.eval()
        # breakpoint()
        model.valid = True
        model.encoder.valid = True
        if model.inject_wav2vec_adapter or model.inject_post_len_adapter:
            samples = split_to_neu_and_exp(sample)
            pass_adapters = [False, True]
            for i, sample in enumerate(samples):
                if sample:
                    with torch.no_grad():
                        loss, sample_size, logging_output, _, _ = criterion(model, sample, step="st",
                                                                            pass_adapter=pass_adapters[i])
        else:
            with torch.no_grad():  # one validation step  for other models
                loss, sample_size, logging_output, _, _ = criterion(model, sample, step="st")
        return loss, sample_size, logging_output

    def get_batch_iterator(
            self,
            dataset,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
            num_workers=0,
            epoch=1,
            data_buffer_size=0,
            disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """

        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        # breakpoint()
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        if self.order_frames:
            with data_utils.numpy_seed(seed):
                print("-------------    order dataset by frames as in tsv ------------")
                # breakpoint()
                indices = dataset.ordered_indices()
        else:
            print("-------------     don't order the whole dataset by frames, as it has been pre-ordered ------------")
            indices = [i for i in range(len(dataset))]
        print("indices: ", indices)
        # breakpoint()
        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(indices, dataset, max_positions, ignore_invalid_inputs)

        self.indices = indices

        def batchify_by_batch_size(indices, bs):
            # assert (len(indices) % 2) == 0, "the number of data must be even"
            # assert (bs % 2) == 0, "batch size must be even"
            batchified_indices = [indices[x:x + bs] for x in range(0, len(indices), bs)]
            return batchified_indices

        # breakpoint()
        # create mini-batches with given size constraints
        if not self.make_even_batch:
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = batchify_by_batch_size(indices, bs=max_sentences)
        # breakpoint()
        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter


@register_task("speech_to_text_reg_same_len", dataclass=SpeechToTextRegTaskConfig)
class SpeechToTextRegSameLenTask(SpeechToTextRegTask):

    def __init__(self, cfg, tgt_dict, label_dict):
        super().__init__(cfg, tgt_dict, label_dict)
        print("turning off data augmentation....")
        self.da_p_augm = 1
        self.da_effects_info = {
            "tempo": list(map(float, "1,1".split(","))),
            "pitch": list(map(int, "0,0".split(","))),
            "echo": {
                "delay": list(map(int, "0,0".split(","))),
                "decay": list(map(float, "0,0".split(",")))
            }
        }

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        if check_paired_neu_exp(sample):
            if self.regularize:  # set regularize to be False as a sanity check for my code that only st is performed.
                print("--------- splitting samples to neu and exp --------")
                self.make_same_length(sample)
                samples = split_to_neu_and_exp(sample)  # always compute neu first !
                steps = ["st", "st-reg"]
            return self._train_step(samples, model, criterion, optimizer, update_num, steps, ignore_grad=ignore_grad)
        else:
            print("neu and exp must be paired. Unpaired data should not be used. Writing this index to output for fixation. ")
            # breakpoint()
            tmp_dic = {"id": sample["id"].tolist(), "mem_index": sample["mem_index"].tolist()}

            self.write_unmatched(tmp_dic)
            self.make_same_length(sample)
            samples = split_to_neu_and_exp(sample)
            steps = ["st", "st-reg"]
            return self._train_step(samples, model, criterion, optimizer, update_num, steps, ignore_grad=ignore_grad)

            # import sys
            # sys.exit()
    def write_unmatched(self, dic):
        f_name = "train_merged_pair_unmatch_ids.json"
        with open(f_name, "a") as f:
            json.dump([dic], f)

    def make_same_length(self, sample):
        # breakpoint()
        neu_indices, exp_indices = retrieve_neu_exp_indices(sample)
        try:
            assert torch.all((neu_indices + 1).eq(exp_indices)), "The neu and exp indices must match!! "
            assert len(neu_indices) == 1, "currently, it only supports a batch size of 2"
        except Exception as e:
            print("e: ", e)
            breakpoint()
        if sample['net_input']['src_lengths'][neu_indices] > sample['net_input']['src_lengths'][exp_indices]:
            pivot_ind = exp_indices
            other_ind = neu_indices
        else:
            pivot_ind = neu_indices
            other_ind = exp_indices
        self.shorten_len_by_pivot_ind(sample, pivot_ind, other_ind)

        assert sample['net_input']['src_tokens'].size()[-1] == sample['net_input']['src_lengths'][pivot_ind].item(), \
            "src_tokens lengths must equal to shorter sample's length"

    @staticmethod
    def shorten_len_by_pivot_ind(sample, pivot_ind, other_ind):
        sample['net_input']['src_lengths'][other_ind] = sample['net_input']['src_lengths'][pivot_ind]

        long_sample = sample['net_input']['src_tokens'][other_ind]
        target_len = sample['net_input']['src_lengths'][pivot_ind].item()

        ind_to_keep = torch.tensor(sorted(random.sample(range(0, long_sample.size()[-1]), target_len))).to(pivot_ind)

        sample['net_input']['src_tokens'][other_ind][:, :target_len] = sample['net_input']['src_tokens'][other_ind][:, ind_to_keep]
        sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'][:, :target_len]


@dataclass
class SpeechToTextRegValTaskConfig(SpeechToTextRegTaskConfig):
    val_data: str = field(
        default="train",
        metadata={"help": "type of data to be evaluated."}
    )
    save_speech_repre: bool = field(
        default=False,
        metadata={"help": "whether to save speech representation"}
    )
    save_enc_out: bool = field(
        default=False,
        metadata={"help": "whether to save encoder output"}
    )
    save_dec_out: bool = field(
        default=False,
        metadata={"help": "whether to save decoder output"}
    )
    agg_mode: str = field(
        default="avg",
        metadata={"help": "pooling method for plotting"}
    )


@register_task("speech_to_text_reg_val", dataclass=SpeechToTextRegValTaskConfig)
class SpeechToTextRegValTask(SpeechToTextRegTask):

    def __init__(self, cfg, tgt_dict, label_dict):
        super().__init__(cfg, tgt_dict, label_dict)
        self.val_data = cfg.val_data
        self.save_speech_repre = cfg.save_speech_repre
        self.save_enc_out = cfg.save_enc_out
        self.save_dec_out = cfg.save_dec_out
        self.agg_mode = cfg.agg_mode

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        return None, 1, {"sample_size": 1, "ntokens": 1}

    def valid_step(self, sample, model, criterion):

        if check_paired_neu_exp(sample):
            samples = split_to_neu_and_exp(sample)

            model.eval()
            exp_neu_dic = {"1": None, "0": None}
            for sample in samples:

                enc_dic = {"id": sample["id"].tolist(), "labels": sample["label"].squeeze(1).tolist(),
                           "target_lengths": sample["target_lengths"].tolist(),
                           "ori_src_lengths": sample["net_input"]["src_lengths"].tolist(),
                           }

                with torch.no_grad():  # one validation step  for other models
                    loss, sample_size, logging_output, enc_out, decoder_net_output = criterion(model, sample, step="st")
                assert len(enc_out["encoder_out"]) == 1, "the length of the list must be 1."
                masks = enc_out["encoder_padding_mask"][0]
                enc_dic["redu_src_lengths"] = self.convert_mask_to_lengths(masks).tolist()
                # breakpoint()

                last_hiddens = enc_out["encoder_out"]

                if self.agg_mode == "avg":
                    enc_pool_out = model.get_avg(last_hiddens, [masks])[0]  # last_hiddens: [S x B x D]; mask: B X S
                elif self.agg_mode == "first":
                    enc_pool_out = model.get_first(last_hiddens)
                else:
                    raise NotImplementedError
                enc_dic["enc_pool_out"] = enc_pool_out.tolist()
                KEY = str(enc_dic["labels"][0])
                exp_neu_dic[KEY] = enc_dic
                # breakpoint()
                if self.save_dec_out:
                    self._save_dec_out(model, sample, decoder_net_output)
            assert exp_neu_dic['1']['target_lengths'] == exp_neu_dic['0'][
                'target_lengths'], "target_lengths must be the same for exp and neu."
            if self.save_enc_out:
                self._write_enc_to_json(exp_neu_dic)
            return loss, sample_size, logging_output
        else:
            return None, 1, {"sample_size": 1, "ntokens": 1}

    def _save_dec_out(self, model, sample, decoder_net_output):
        # breakpoint()
        assert len(sample['id']) == 1, "Currently can only process one item at a time."

        print(f"sample[target]: {sample['target']}")
        dec_att_dic = {"id": sample["id"].tolist(), "target_lengths": sample["target_lengths"].item()}
        x = decoder_net_output[1]['x'].cpu().numpy().tolist()
        inner_states = decoder_net_output[1]['inner_states'][1:]
        inner_states = [item.tolist() for item in inner_states]
        last_cross_attns_all_heads = decoder_net_output[1]['attn'][0].cpu().numpy().tolist() #[0]

        dec_att_dic["x"] = x
        # last_cross_attns_all_heads
        dec_att_dic["inner_states"] = inner_states
        dec_att_dic["last_cross_attns"] = last_cross_attns_all_heads
        f_name = f"{self.val_data}.x.last.cros.attn.all.inn.state.json"
        with open(f_name, "a") as f:
            json.dump(dec_att_dic, f)
            f.write(os.linesep)

        # print(type(decoder_net_output))

    def _write_enc_to_json(self, enc_dic):
        all_enc_pool_out = []
        all_labels = []
        all_ids = []
        # breakpoint()
        for key, value in enc_dic.items():
            enc_pool_out = value.pop("enc_pool_out")
            # enc_pool_out = [",".join([str(o) for o in out])  for item in value.pop("enc_pool_out") for out in item]
            enc_pool_out = [",".join([str(o) for o in out]) for out in enc_pool_out]

            all_enc_pool_out.extend(enc_pool_out)

            label = [str(l) for l in value["labels"]]
            all_labels.extend(label)

            ids = [str(i) for i in value["labels"]]
            all_ids.extend(ids)
        # breakpoint()
        # all_labels = np.array(list(map(float, all_labels)))
        # all_enc_pool_out = np.array(list(map(float, all_enc_pool_out)))
        if not self.save_speech_repre:
            with open(f"{self.val_data}.json", "a") as f:
                json.dump(enc_dic, f)
                f.write(os.linesep)

        else:
            vector_file = f"{self.val_data}.spe.vector.{self.agg_mode}.txt"
            label_file = f"{self.val_data}.label.{self.agg_mode}.txt"
            id_file = f"{self.val_data}.id.{self.agg_mode}.txt"
            self.write_to_text(vector_file, all_enc_pool_out)
            self.write_to_text(label_file, all_labels)
            self.write_to_text(id_file, all_ids)

        # vector_file = "train_merged.spe.vector.txt"
        # label_file = "train_merged.label.txt"
        # all_enc_pool_out = np.genfromtxt(vector_file, delimiter=",")
        # all_labels = np.genfromtxt(label_file, delimiter=",")
        # from sklearn.cluster import KMeans
        # kmeans = KMeans(n_clusters=2, random_state=0).fit(all_enc_pool_out)

    @staticmethod
    def write_to_text(f_name, lst):
        lst = [item + "\n" for item in lst]
        f = open(f_name, "a")
        f.writelines(lst)
        f.close()

    @staticmethod
    def convert_mask_to_lengths(masks):
        return (masks == False).sum(1)


@dataclass
class SpeechToTextContTaskConfig(SpeechToTextRegTaskConfig):
    contrastive: bool = field(
        default=True,
        metadata={"help": "for contrastive st"}
    )
    warmup_steps_contrastive: int = field(
        default=5,
        metadata={"help": "warmup steps before contrastive, so that feature memory has a good initialisation."}
    )
    similarity_method: str = field(
        default="bleu",
        metadata={"help": "similarity method for finding top k most similar hard negatives as prior."}
    )
    sim_cutoff: float = field(
        default=95.0,
        metadata={"help": "cut off score for selecting negative examples"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to convert samples to fp16 when updating feature memory"}
    )


@register_task("speech_to_text_contrastive", dataclass=SpeechToTextContTaskConfig)
class SpeechToTextContTask(SpeechToTextRegTask):

    def __init__(self, cfg, tgt_dict, label_dict):
        SpeechToTextModTask.__init__(self, cfg, tgt_dict, label_dict)
        self.contrastive = cfg.contrastive
        self.warmup_steps_contrastive = cfg.warmup_steps_contrastive
        self.sim_method = cfg.similarity_method
        self.sim_cutoff = cfg.sim_cutoff
        self.fp16 = cfg.fp16
        self.order_frames = cfg.order_frames

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):

        print(f"------------------------------- update_num: {update_num} ---------------------")
        if check_paired_neu_exp(
                sample) and model.index_memory_updated.item() and self.contrastive:  # set contrastive to be False as a sanity check for my code that only st is performed.
            print(
                "--------- splitting samples to neu and exp --------")  # model.index_memory_updated as a signal to do contrastive learning
            samples = split_to_neu_and_exp(sample)
            steps = ["st", "st-contr"]
        else:
            samples = split_to_half(sample)
            steps = ["st", "st"]
        if not model.index_memory_updated.item() and update_num == self.warmup_steps_contrastive:
            print(" !!!!!!!!!!!! below function can only be called once !!!!!!!!!!!!!")
            print(" update_index_feature_memory ")
            self.one_off_update_index_feature_memory(model)
        return self._train_step(samples, model, criterion, optimizer, update_num, steps, ignore_grad=False)

    # def _train_step(self, samples, model, criterion, optimizer, update_num, steps, ignore_grad=False):
    #     total_loss, total_sample_size, total_logging_output = 0, 0, Counter()
    #     sample_size = None
    #     enc_pool_out = None
    #
    #     for step, sample in zip(steps, samples):
    #         print("sample id & label: ", sample["id"], sample["label"])
    #         print("step to be completed: ", step)
    #         prev_sample_size = sample_size
    #         model.train()
    #         model.set_num_updates(update_num)
    #         # breakpoint()
    #
    #         with torch.autograd.profiler.record_function("forward"):
    #             loss, sample_size, logging_output, enc_pool_out = criterion(model, sample, step=step,
    #                                                                         prev_enc_pool_out=enc_pool_out)
    #         # breakpoint()
    #         if ignore_grad:
    #             loss *= 0
    #         with torch.autograd.profiler.record_function("backward"):
    #             optimizer.backward(loss)
    #         total_loss += loss
    #         # del loss
    #         total_sample_size += sample_size
    #         total_logging_output = total_logging_output + Counter(logging_output)
    #         # breakpoint()
    #     if step and step == "st-contr":
    #         assert prev_sample_size == sample_size, "Two mini-batches must have the same target tokens."
    #         # breakpoint()
    #         assert torch.unique(sample["label"], dim=0).item() == 1, "The second mini-batch must be expressive with labels being 1"
    #     # breakpoint()
    #     return total_loss, total_sample_size, dict(total_logging_output)

    def one_off_update_index_feature_memory(self, model):
        model.eval()
        all_exp_tgt_str = []
        train_data = []
        for dataset in self.datasets:
            if dataset.startswith("train"):
                train_data.append(dataset)
                dataset = self.datasets[dataset]
                # breakpoint()
                try:
                    # exp_tgt = [ ]
                    exp_tgt = (
                        lambda dataset=dataset: [dataset[i][2][1:].numpy().tolist() for i in range(len(dataset)) if
                                                 dataset[i][3].item()])()
                    assert len(exp_tgt) < len(dataset), "number of exp must be less then the number of exp+neu"
                except Exception as e:
                    print("e: ", e)
                    breakpoint()
                    exp_tgt = (
                        lambda dataset=dataset: [dataset[i][2][1:].numpy().tolist() for i in range(len(dataset))])()
                exp_tgt_str = [" ".join([str(int_) for int_ in lst]) for lst in exp_tgt]
                all_exp_tgt_str += exp_tgt_str  # this is in order.

        def update_index_memory_with_topk_sim_pairs(str_lst,
                                                    k):  # exclude too similar pairs, eg those have the same translation.
            if k > len(str_lst):
                k = len(str_lst)
            for i, one_str in enumerate(str_lst):
                sim_scores = np.array([0.0 for _ in range(len(str_lst))])  # TODO: remove for loop
                # breakpoint()
                # sim_scores = np.array([ sacrebleu.sentence_bleu(one_str, [oth_str]).score for oth_str in str_lst])
                # sim_scores =  np.array([ np.array([sacrebleu.sentence_bleu(one_str, [str_lst[i]]).score for i in range(len(str_lst))])])
                for j, oth_str in enumerate(str_lst):
                    score = sacrebleu.sentence_bleu(one_str, [oth_str]).score
                    if score < self.sim_cutoff:
                        sim_scores[j] = score

                # sim_scores_arr = np.asarray(sim_scores)
                topk_ind = torch.as_tensor(np.argpartition(sim_scores, -k)[-k:])
                # breakpoint()
                index_memory.update(topk_ind, i)

        # if not model.index_memory_updated.item(): # only update the index_memory once!!
        print("........  updating index memory  .......")
        # update index memory
        # breakpoint()
        index_memory = model.index_memory
        k = index_memory.shape()[1]
        update_index_memory_with_topk_sim_pairs(all_exp_tgt_str, k)
        model.index_memory_updated[0] = 1

        # update feature_memory
        feature_memory = model.feature_memory
        epoch_iter = list(self.dataset_to_epoch_iter.values())[0]
        assert isinstance(epoch_iter, iterators.EpochBatchIterator)
        itr = epoch_iter._get_iterator_for_epoch(0, shuffle=False)  # reuse the epoch iter to emit samples

        progress = progress_bar.progress_bar(
            itr,
            epoch=0,
            prefix=f"valid on '{train_data}' subset",
            default_log_format="tqdm",
        )
        start = 0
        # sample is emitted in order as the tsv file
        # although within a sample batch, the batch is ordered by # of frames in descending order
        # enc_pool_out array need to reversed.
        print("Updating feature memory ....... ")
        for i, sample in enumerate(progress):
            # breakpoint()
            sample, is_dummy_batch = self._prepare_sample(sample, device=model.device)
            sample_lst = split_to_neu_and_exp(sample)
            assert len(sample_lst) == 2, "The length should be 2, although the first item may be empty!!"
            exp_sample = sample_lst[1]
            if not exp_sample:
                continue
            n = exp_sample["nsentences"]
            ids = exp_sample["id"]
            print("*" * 50)
            print("ids, labels:", ids, exp_sample["label"])
            # print("*" * 50)
            assert n == len(ids)
            model.eval()
            with torch.no_grad():
                encoder_out = model.encoder(**exp_sample["net_input"])

            last_hiddens = encoder_out["encoder_out"]
            encoder_masks = encoder_out["encoder_padding_mask"]
            enc_pool_out = model.get_avg(last_hiddens, encoder_masks)[0]
            # breakpoint()  # should flip; or, don't need start to index, use mem_index within exp_example
            enc_pool_out = torch.flip(enc_pool_out, [0])
            feature_memory.continuous_update(enc_pool_out, start, n)
            start += n
        print("Updating feature memory completed! ")

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output, _ = criterion(model, sample, step="st")
        return loss, sample_size, logging_output

    def _prepare_sample(self, sample, device="cuda"):
        if sample == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )

        if sample is None or len(sample) == 0:
            assert (
                    self._dummy_batch is not None and len(self._dummy_batch) > 0
            ), "Invalid dummy batch: {}".format(self._dummy_batch)
            sample, _ = self._prepare_sample(self._dummy_batch, is_dummy=True)
            return sample, True

        if device.type == "cuda":
            sample = utils.move_to_cuda(sample)

        def apply_half(t):
            if t.dtype is torch.float32:
                return t.half()
            return t

        if self.fp16:
            sample = utils.apply_to_sample(apply_half, sample)

        return sample, False


@register_task("speech_to_text_iwslt21_multitask", dataclass=SpeechToTextModTaskConfig)
class SpeechToTextModMulTask(SpeechToTextModTask):

    def __init__(self, cfg, tgt_dict, label_dict):
        super().__init__(cfg, tgt_dict, label_dict)

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        step = self.get_task(update_num)
        # print("update_num after: ", update_num)

        with torch.autograd.profiler.record_function("forward"):
            # breakpoint()
            loss, sample_size, logging_output = criterion(model, sample, step=step)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        # breakpoint()
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, step="both")
        return loss, sample_size, logging_output

    def discrim_step(self, num_updates):
        return num_updates % 2 == 1

    def get_task(self, num_updates):
        return "discriminator" if self.discrim_step(num_updates) else "translator"
