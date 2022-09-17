# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import logging
import os.path as op
import re
from typing import Dict, List, Optional, Tuple
from random import gauss
import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.audio_utils import get_fbank, get_waveform
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform

import json
import os
import sacrebleu
import json
import random
import string
import os
import difflib
import soundfile as sf
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class S2TDataConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for " "S2T data config")
        self.config = {}
        # breakpoint()
        if op.isfile(yaml_path):
            try:
                with open(yaml_path) as f:
                    self.config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                logger.info(f"Failed to load config from {yaml_path}: {e}")
        else:
            logger.info(f"Cannot find {yaml_path}")

    @property
    def vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("vocab_filename", "dict.txt")

    @property
    def spoken_vocab_path(self):
        return self.config.get("spoken_vocab_path", "")

    @property
    def spoken_vocab_fname(self):
        return self.config.get("spoken_vocab_fname", "")

    @property
    def spoken_vocab_audio_path(self):
        return self.config.get("spoken_vocab_audio_path", "")

    @property
    def spoken_vocab_ben_audio_path(self):
        return self.config.get("spoken_vocab_ben_audio_path", "")

    @property
    def en_ben_dict_path(self):
        return self.config.get("en_ben_dict_path", "")

    @property
    def en_ben_dict(self):
        return self.config.get("en_ben_dict", "")

    @property
    def code_mix_prob(self):
        return self.config.get("code_mix_prob", 0)

    @property
    def cw_frequency(self):
        return self.config.get("cw_frequency", 1)

    @property
    def shuffle(self) -> bool:
        """Shuffle dataset samples before batching"""
        return self.config.get("shuffle", False)

    @property
    def pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("pre_tokenizer", {"tokenizer": None})

    @property
    def bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("bpe_tokenizer", {"bpe": None})

    @property
    def prepend_tgt_lang_tag(self) -> bool:
        """Prepend target lang ID token as the target BOS (e.g. for to-many
        multilingual setting). During inference, this requires `--prefix-size 1`
        to force BOS to be lang ID token."""
        return self.config.get("prepend_tgt_lang_tag", False)

    @property
    def prepend_src_lang_tag(self) -> bool:
        """Prepend source lang ID token as the source EOS (e.g. for to-many
        multilingual setting); for mBart-50 only (other mBarts have different
        configurations) """
        return self.config.get("prepend_src_lang_tag", False)

    @property
    def input_feat_per_channel(self):
        """The dimension of input features (per audio channel)"""
        return self.config.get("input_feat_per_channel", 80)

    @property
    def input_channels(self):
        """The number of channels in the input audio"""
        return self.config.get("input_channels", 1)

    @property
    def sampling_alpha(self):
        """Hyper-parameter alpha = 1/T for temperature-based resampling.
        (alpha = 1 for no resampling)"""
        return self.config.get("sampling_alpha", 1.0)

    @property
    def use_audio_input(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("use_audio_input", False)

    @property
    def mix_speech(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("mix_speech", False)

    @property
    def alpha(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("alpha", 0.05)

    @property
    def gauss_alpha(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("gauss_alpha", False)

    @property
    def gauss_mean(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("gauss_mean", 0.1)

    @property
    def gauss_st(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("gauss_st", 0.2)

    @property
    def whitenoise(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("whitenoise", False)

    @property
    def audio_root(self):
        """Audio paths in the manifest TSV can be relative and this provides
        the root path. Set this to empty string when using absolute paths."""
        return self.config.get("audio_root", "")

    @property
    def no_feature_transforms_on_stitched_voice(self):
        return self.config.get("no_feature_transforms_on_stitched_voice", False)

    @property
    def num_pseudo_spk(self):
        return self.config.get("num_pseudo_spk", 0)

    @property
    def mis_align(self):
        return self.config.get("mis_align", False)

    def get_feature_transforms(self, split, is_train):
        """Split-specific feature transforms. Allowing train set wildcard `_train`,
        evaluation set wildcard `_eval` and general wildcard `*` for matching."""
        from copy import deepcopy

        cfg = deepcopy(self.config)
        _cur = cfg.get("transforms", {})
        cur = _cur.get(split)
        cur = _cur.get("_train") if cur is None and is_train else cur
        cur = _cur.get("_eval") if cur is None and not is_train else cur
        cur = _cur.get("*") if cur is None else cur
        cfg["transforms"] = cur
        return cfg

    @property
    def order_within_batch(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("order_within_batch", True)

    @property
    def save_extra_tokenized_target(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("save_extra_tokenized_target", False)


def is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78


def is_flac_or_wav_data(data: bytes) -> bool:
    is_flac = data[0] == 102 and data[1] == 76
    is_wav = data[0] == 82 and data[1] == 73
    return is_flac or is_wav


def read_from_uncompressed_zip(file_path, offset, file_size) -> bytes:
    with open(file_path, "rb") as f:
        f.seek(offset)
        data = f.read(file_size)
    return data


def get_features_from_npy_or_audio(path):
    ext = op.splitext(op.basename(path))[1]
    if ext not in {".npy", ".flac", ".wav"}:
        raise ValueError(f'Unsupported file format for "{path}"')
    return np.load(path) if ext == ".npy" else get_fbank(path)


def get_features_or_waveform_from_uncompressed_zip(
        path, byte_offset, byte_size, need_waveform=False
):
    assert path.endswith(".zip")
    data = read_from_uncompressed_zip(path, byte_offset, byte_size)
    f = io.BytesIO(data)
    if is_npy_data(data):
        features_or_waveform = np.load(f)
    elif is_flac_or_wav_data(data):
        features_or_waveform = \
            get_waveform(f, always_2d=False)[0] if need_waveform else get_fbank(f)
    else:
        raise ValueError(f'Unknown file format for "{path}"')
    return features_or_waveform


def get_features_or_waveform(path: str, need_waveform=False):
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        need_waveform (bool): return waveform instead of features.

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    _path, *extra = path.split(":")
    if not op.exists(_path):
        raise FileNotFoundError(f"File not found: {_path}")

    if len(extra) == 0:
        if need_waveform:
            waveform, _ = get_waveform(_path, always_2d=False)
            return waveform
        return get_features_from_npy_or_audio(_path)
    elif len(extra) == 2:
        extra = [int(i) for i in extra]
        if need_waveform:
            waveform, _ = get_waveform(_path, start=extra[0], frames=extra[1],
                                       always_2d=False)
            return waveform
        features_or_waveform = get_features_or_waveform_from_uncompressed_zip(
            _path, extra[0], extra[1], need_waveform=need_waveform
        )
    else:
        raise ValueError(f"Invalid path: {path}")

    return features_or_waveform


def _collate_frames(
        frames: List[torch.Tensor], is_audio_input: bool = False
) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out


spec_tokens = "@$%"
punctuation = string.punctuation
for tok in spec_tokens:
    punctuation = punctuation.replace(tok, "")
file_type = "wav"
default_tok = "a"


def normalize(sent):
    sent = sent.strip("\n")
    sent = sent.lower()
    sent = sent.translate(str.maketrans('', '', punctuation))
    for tok in spec_tokens:
        sent = sent.replace(tok, f" {tok} ")
    return sent


def stitch_voice(sent, vocab, voice_path):
    sent = normalize(sent)
    tokens = [tok for tok in sent.split() if tok]
    voice_fpaths = []
    for tok in tokens:
        fname = tok + "." + file_type
        voice_fpath = os.path.join(voice_path, fname)
        is_exist = os.path.exists(voice_fpath)
        if not is_exist:
            sim_toks = difflib.get_close_matches(tok, vocab, n=1)
            if sim_toks:
                sim_tok = sim_toks[0]
                fname = sim_tok + "." + file_type
            else:
                fname = default_tok + "." + file_type
            voice_fpath = os.path.join(voice_path, fname)
            try:
                assert os.path.exists(voice_fpath), "voice file path must exit; something went wrong!"
            except Exception as e:
                print("error: ", e)
                breakpoint()
        voice_fpaths.append(voice_fpath)
    return voice_fpaths


def smooth_stitched_speech(waveform_list, crossfade=100):
    # 1.5 second crossfade
    if len(waveform_list) == 1:
        return waveform_list[0]
    elif len(waveform_list) == 0:
        raise Exception("waveform_list cannot be empty!!!")
    waveform = waveform_list[0]
    for each in waveform_list[1:]:
        waveform = waveform.append(each, crossfade)
    return waveform


def generate_stitched_voice(sent, vocab, voice_path, crossfade=100):
    assert sent, "sent cannot be empty"
    voice_fpaths = stitch_voice(sent, vocab, voice_path)
    waveform_list = []
    for file in voice_fpaths:
        waveform = AudioSegment.from_file(file, format="wav")
        waveform_list.append(waveform)
    stitched_waveform = smooth_stitched_speech(waveform_list, crossfade)
    return stitched_waveform


class SpeechToTextDataset(FairseqDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
            self,
            split: str,
            is_train_split: bool,
            data_cfg: S2TDataConfig,
            audio_paths: List[str],
            n_frames: List[int],
            src_texts: Optional[List[str]] = None,
            tgt_texts: Optional[List[str]] = None,
            labels: Optional[List[str]] = None,
            mem_indices: Optional[List[str]] = None,
            speakers: Optional[List[str]] = None,
            src_langs: Optional[List[str]] = None,
            tgt_langs: Optional[List[str]] = None,
            ids: Optional[List[str]] = None,
            tgt_dict: Optional[Dictionary] = None,
            pre_tokenizer=None,
            bpe_tokenizer=None,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.data_cfg = data_cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert labels is None or len(labels) == self.n_samples
        assert mem_indices is None or len(mem_indices) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
                tgt_dict is not None and tgt_texts is not None
        )
        assert (tgt_dict is None and labels is None) or (
                tgt_dict is not None and labels is not None
        )

        self.src_texts, self.tgt_texts, self.labels, self.mem_indices = src_texts, tgt_texts, labels, mem_indices
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.tgt_dict = tgt_dict
        # breakpoint()
        self.check_tgt_lang_tag()
        # self.check_src_lang_tag()
        self.ids = ids
        self.shuffle = data_cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms(split, is_train_split)
        )

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer

        self.order_within_batch = data_cfg.order_within_batch
        self.save_extra_tokenized_target = data_cfg.save_extra_tokenized_target

        # self.build_target_sim_index()
        if self.data_cfg.spoken_vocab_path and self.data_cfg.spoken_vocab_fname and self.data_cfg.spoken_vocab_audio_path:
            self.spoken_vocab_path, self.spoken_vocab_fname, self.spoken_vocab_audio_path = \
                self.data_cfg.spoken_vocab_path, self.data_cfg.spoken_vocab_fname, self.data_cfg.spoken_vocab_audio_path
            self.build_spoken_vocab()
            # if self.spoken_vocab_path and self.spoken_vocab_fname and self.spoken_vocab_audio_path:

        if self.data_cfg.spoken_vocab_ben_audio_path:
            assert self.data_cfg.en_ben_dict_path and self.data_cfg.en_ben_dict
            # self.build_ben_spoken_vocab()
            self.build_en_ben_dict()

        logger.info(self.__repr__())

    def __repr__(self):
        return (
                self.__class__.__name__
                + f'(split="{self.split}", n_samples={self.n_samples}, '
                  f"prepend_tgt_lang_tag={self.data_cfg.prepend_tgt_lang_tag}, "
                  f"shuffle={self.shuffle}, transforms={self.feature_transforms})"
        )

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace("{}", "(.*)")
        return re.match(pattern, token)

    def build_target_sim_index(self, k=5):
        breakpoint()
        if self.data_cfg.mix_speech:
            print("building target similarity index .......")
            self.tgt_sim_index = {}
            L = len(self.tgt_texts)
            for index in range(L):
                print("index: ", index)
                tgt = self.tgt_texts[index]
                sim_scores = np.array([sacrebleu.sentence_bleu(t, [ref], smooth_method='exp').score for t, ref in
                                       zip([tgt] * len(self.tgt_texts), self.tgt_texts)])
                top_k_index = np.argpartition(sim_scores, -k)[-k:]
                self.tgt_sim_index[index] = top_k_index
        breakpoint()

    def check_tgt_lang_tag(self):
        if self.data_cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)

    def check_src_lang_tag(self):
        if self.data_cfg.prepend_src_lang_tag:
            assert self.src_langs is not None and self.tgt_dict is not None
            src_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.src_langs)
            ]
            assert all(t in self.tgt_dict for t in src_lang_tags)

    def tokenize_text(self, text: str):
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text)
        return text

    def build_spoken_vocab(self):
        def read_file(path, fname):
            with open(os.path.join(path, fname), "r") as f:
                data = f.readlines()
            return [item.strip("\n") for item in data]

        self.spoken_vocab = read_file(self.spoken_vocab_path, self.spoken_vocab_fname)

    def build_en_ben_dict(self):
        import csv
        def read_csv_file(path, fname):
            with open(os.path.join(path, fname)) as file:
                csvreader = csv.reader(file)
                data = [item[:2] for item in csvreader][1:]
            data_dic = {item[0]: item[1] for item in data}
            return data_dic

        self.en_ben_dict = read_csv_file(self.data_cfg.en_ben_dict_path, self.data_cfg.en_ben_dict)

    def __getitem__(
            self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        tmp_text = None
        if self.audio_paths[index].startswith("[spoken_vocab]"):
            if self.data_cfg.num_pseudo_spk > 0:
                spk = random.choice(range(self.data_cfg.num_pseudo_spk))
                temp_path = self.spoken_vocab_audio_path.split("audio")
                if spk != 0:
                    self.spoken_vocab_audio_path = temp_path[0] + f"audio-{str(spk)}/wav24k"
                else:
                    self.spoken_vocab_audio_path = temp_path[0] + f"audio/wav24k"

            text = self.audio_paths[index]
            text = text.split("[spoken_vocab]")[1]

            if self.data_cfg.en_ben_dict:
                random.seed(index)

                def mix_code(txt):
                    en_words = (self.en_ben_dict.keys())
                    matched = list(filter(lambda x: " "+x+" " in txt, en_words))
                    if matched:
                        for _ in range(0, self.data_cfg.cw_frequency):
                            one_en_tok = random.choice(matched)
                            # print("matched word in en-ben-dict: ", one_en_tok)
                            one_ben_tok = self.en_ben_dict[one_en_tok]
                            # print("sentence before code-mixing: ", txt)
                            txt = txt.replace(" "+one_en_tok+" ", " "+one_ben_tok+" ")
                            # print("sentence after code-mixing: ", txt)
                    return txt

                if random.random() < self.data_cfg.code_mix_prob:
                    text = mix_code(text)

            source = generate_stitched_voice(text, self.spoken_vocab, self.spoken_vocab_audio_path)
            source = np.asarray(source.get_array_of_samples())

            if not self.data_cfg.no_feature_transforms_on_stitched_voice:
                if self.feature_transforms is not None:
                    assert not self.data_cfg.use_audio_input
                    source = self.feature_transforms(source)

            tmp_text = text
        else:
            source = get_features_or_waveform(
                self.audio_paths[index], need_waveform=self.data_cfg.use_audio_input
            )
            if self.feature_transforms is not None:
                assert not self.data_cfg.use_audio_input
                source = self.feature_transforms(source)

        if self.data_cfg.mis_align and tmp_text:
            # print("1" * 10)
            # prev_txt = tmp_text[:10]
            next_txt = tmp_text[-10:]
            # if not prev_txt:
            #     prev_txt = "hmm"
            if not next_txt:
                next_txt = "hmm"
            # prev_source = generate_stitched_voice(prev_txt, self.spoken_vocab, self.spoken_vocab_audio_path)
            # prev_source = np.asarray(prev_source.get_array_of_samples())
            next_source = generate_stitched_voice(next_txt, self.spoken_vocab, self.spoken_vocab_audio_path)
            next_source = np.asarray(next_source.get_array_of_samples())
            # source = np.concatenate((prev_source, source, next_source), axis=None)
            source = np.concatenate((next_source, source), axis=None)

        if self.data_cfg.mix_speech and self.is_train_split:
            # mix_index = np.random.choice(self.tgt_sim_index[index], 1)
            if not self.data_cfg.whitenoise:
                mix_index = random.choice(range(len(self.audio_paths)))
                mix_source = get_features_or_waveform(self.audio_paths[mix_index],
                                                      need_waveform=self.data_cfg.use_audio_input)
            else:
                print("white noise.....")
                mix_source = np.random.normal(0, source.std(), source.size)
            if self.feature_transforms is not None:
                assert not self.data_cfg.use_audio_input
                mix_source = self.feature_transforms(mix_source)

            if len(mix_source) > len(source):
                mix_source = mix_source[:len(source)]
                # source = np.concatenate([source, mix_source[len(source)]])
            else:
                mix_source = np.concatenate([mix_source, source[len(mix_source):]])
            if self.data_cfg.gauss_alpha:
                alpha = gauss(self.data_cfg.gauss_mean, self.data_cfg.gauss_st)
            alpha = self.data_cfg.alpha
            source = (1 - alpha) * source + alpha * mix_source
        source = torch.from_numpy(source).float()

        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()

            # breakpoint()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)
            # print(self.is_train_split)
            # breakpoint()
            if self.save_extra_tokenized_target:  # only do this in the validation mode
                print("target tokenized string: ", tokenized)
                print("number of target tokens: ", len(tokenized.split(" ")))
                print("target: ", target)
                # assert target.size()[0] == len(tokenized.split(" ")) + 2, "the length must be equal; the former has eos added at the end."
                self._save_extra_tokenized_target(index, tokenized, lang_tag, target)

        if self.src_texts is not None:
            src_tokenized = self.tokenize_text(self.src_texts[index])
            transcript = self.tgt_dict.encode_line(
                src_tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            # print("source tokenzied: ", src_tokenized)
            if self.data_cfg.prepend_src_lang_tag:
                # print("------ adding lang tok to transcript -----")
                src_lang_tag = self.LANG_TAG_TEMPLATE.format(self.src_langs[index])
                src_lang_tag_idx = self.tgt_dict.index(src_lang_tag)
                transcript = torch.cat((torch.LongTensor([src_lang_tag_idx]), transcript), 0)

        # print("taregt: ", tokenized)
        label = None
        # breakpoint()
        if self.labels is not None:
            # rint("getting labels")
            # print("__file__:", __file__)
            tokenized = self.tokenize_text(self.labels[index])
            label = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=False
            ).long()
            if self.labels[index] == "0":
                label[0] = 0
            elif self.labels[index] == "1":
                label[0] = 1
            else:
                raise Exception("Label must be either 0 or 1")
        # print(index, source, target, label, transcript)
        # print("src_tokenized: ", src_tokenized)
        mem_index = None
        if self.mem_indices is not None:
            if self.mem_indices[index]:
                mem_index = torch.tensor([int(self.mem_indices[index])])
            else:
                mem_index = torch.tensor([int(SpeechToTextDatasetCreator.DEFAULT_MEM_INDEX)])
        return index, source, target, label, transcript, mem_index

    def _save_extra_tokenized_target(self, index, tok_string, lang_str, tgt_indices):
        # print(f"tgt_indices 00000000: {tgt_indices}")
        tgt_dic = {"id": index.item(), "target_lengths": len(tgt_indices)}

        if self.data_cfg.prepend_tgt_lang_tag:
            tok_string = lang_str + " " + tok_string
        tok_string += " EOS"
        tok_lst = tok_string.split(" ")
        assert len(tgt_indices) == len(tok_lst), "the lengths must be the same. "  # sanity check
        tgt_dic["tgt_toks"] = tok_lst
        f_name = f"{self.split}.ext.tok.tgt.json"
        print(f"saving tokenized target to output file {f_name}....")
        # breakpoint()
        with open(f_name, "a") as f:
            json.dump(tgt_dic, f)
            f.write(os.linesep)

    def __len__(self):
        return self.n_samples

    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict:
        # print("---  order batch by the number of frames -----")
        # print("number of samples in this batch: ", len(samples))
        # breakpoint()
        if len(samples) == 0:
            return {}

        indices = torch.tensor([i for i, _, _, _, _, _ in samples], dtype=torch.long)
        frames = _collate_frames(
            [s for _, s, _, _, _, _ in samples], self.data_cfg.use_audio_input
        )
        # breakpoint()
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _, _, _, _ in samples], dtype=torch.long)
        # breakpoint()
        if self.data_cfg.order_within_batch:
            # print("ordering within a batch by the actual number of frames")
            n_frames, order = n_frames.sort(descending=True)
        else:
            order = torch.tensor([i for i in range(len(n_frames))], dtype=n_frames.dtype, device=n_frames.device)
        # print("order: ", order)
        # breakpoint()

        indices = indices.index_select(0, order)
        # print("ordered_indices within a batch ", indices)
        # breakpoint()

        frames = frames.index_select(0, order)

        transcript, transcript_lengths = None, None
        transcript_ntokens = None
        if self.src_texts is not None:
            transcript = fairseq_data_utils.collate_tokens(
                [ts for _, _, _, _, ts, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            transcript = transcript.index_select(0, order)
            transcript_lengths = torch.tensor(
                [ts.size(0) for _, _, _, _, ts, _ in samples], dtype=torch.long
            ).index_select(0, order)
            transcript_ntokens = sum(ts.size(0) for _, _, _, _, ts, _ in samples)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        # breakpoint()
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, t, _, _, _ in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t, _, _, _ in samples)

        label = None
        if self.labels is not None:
            label = fairseq_data_utils.collate_tokens([l for _, _, _, l, _, _ in samples], self.tgt_dict.pad())
            # print("original label: \n", label)
            label = label.index_select(0, order)
            # print("re-ordered label: \n", label)

        mem_index = None
        if self.mem_indices is not None:
            mem_index = fairseq_data_utils.collate_tokens([m for _, _, _, _, _, m in samples], self.tgt_dict.pad())
            # print("original mem_index: \n", mem_index)
            mem_index = mem_index.index_select(0, order)
            # print("re-ordered mem_index: \n", mem_index)

        # breakpoint()

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
                "transcript_tokens": transcript,
                "transcript_lengths": transcript_lengths,
                "transcript_ntokens": transcript_ntokens
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
            "label": label,
            "mem_index": mem_index
        }
        # print("out: ", out)
        # breakpoint()
        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            t_len = len(tokenized.split(" "))
        return self.n_frames[index], t_len

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        # breakpoint()
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False


class SpeechToTextDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_TGT_TEXT = "tgt_text"
    KEY_LABEL = "label"
    MEM_INDEX = "mem_index"
    # optional columns
    KEY_SPEAKER, KEY_SRC_TEXT = "speaker", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""
    DEFAULT_LABEL = "0"
    DEFAULT_MEM_INDEX = "-100"

    @classmethod
    def _from_list(
            cls,
            split_name: str,
            is_train_split,
            samples: List[List[Dict]],
            data_cfg: S2TDataConfig,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
    ) -> SpeechToTextDataset:
        audio_paths, n_frames, src_texts, tgt_texts, ids, labels, mem_indices = [], [], [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []

        # breakpoint()
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
            )
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            labels.extend([ss.get(cls.KEY_LABEL, cls.DEFAULT_LABEL) for ss in s])
            mem_indices.extend([ss.get(cls.MEM_INDEX, cls.DEFAULT_MEM_INDEX) for ss in s])

            src_texts.extend(
                [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
            )
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])

        # print("ids: ", ids)
        # print("speakers: ", speakers)
        # breakpoint()
        return SpeechToTextDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            labels,
            mem_indices,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,

        )

    @classmethod
    def _get_size_ratios(cls, ids: List[str], sizes: List[int], alpha: float = 1.0):
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""
        _sizes = np.array(sizes)
        prob = _sizes / _sizes.sum()
        smoothed_prob = prob ** alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

        o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"original sampling probability: {o_str}")
        p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"balanced sampling probability: {p_str}")
        sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
        logger.info(f"balanced sampling size ratio: {sr_str}")
        return size_ratio.tolist()

    @classmethod
    def from_tsv(
            cls,
            root: str,
            data_cfg: S2TDataConfig,
            splits: str,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split: bool,
            epoch: int,
            seed: int,
    ) -> SpeechToTextDataset:
        samples = []
        _splits = splits.split(",")
        for split in _splits:
            tsv_path = op.join(root, f"{split}.tsv")
            if not op.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(
                    f,
                    delimiter="\t",
                    quotechar=None,
                    doublequote=False,
                    lineterminator="\n",
                    quoting=csv.QUOTE_NONE,
                )
                samples.append([dict(e) for e in reader])
                assert len(samples) > 0
        # print("samples: ",samples)
        # breakpoint()
        datasets = [
            cls._from_list(
                name,
                is_train_split,
                [s],
                data_cfg,
                tgt_dict,
                pre_tokenizer,
                bpe_tokenizer,
            )
            for name, s in zip(_splits, samples)
        ]

        if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
            print("----   resampling ----")
            # temperature-based sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]
        return ConcatDataset(datasets)
