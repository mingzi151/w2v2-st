#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive
from tqdm import tqdm


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "duration_ms", "n_frames", "tgt_text", "speaker"]

WAV_MESSAGE = \
    'for f in ${COVOST_ROOT}/*/clips/*.mp3; do\n' \
    '    ffmpeg -i $f -ar 16000 -hide_banner -loglevel error "${f%.mp3}.wav" && rm $f\n' \
    'done\n' \
    "sed 's/\.mp3\t/\.wav\t/g' ${COVOST_ROOT}/**/*.tsv"


class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).

    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str, optional): target (text) language,
        None for no translation (default: None)
        version (int, optional): CoVoST version. (default: 2)
        download (bool, optional): Whether to download the dataset if it is not
        found at root path. (default: ``False``).
    """

    COVOST_URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/covost/"
        "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    )

    VERSIONS = {2}
    SPLITS = ["train", "dev", "test"]

    XX_EN_LANGUAGES = {
        1: ["fr", "de", "nl", "ru", "es", "it", "tr", "fa", "sv-SE", "mn", "zh-CN"],
        2: [
            "fr",
            "de",
            "es",
            "ca",
            "it",
            "ru",
            "zh-CN",
            "pt",
            "fa",
            "et",
            "mn",
            "nl",
            "tr",
            "ar",
            "sv-SE",
            "lv",
            "sl",
            "ta",
            "ja",
            "id",
            "cy",
        ],
    }
    EN_XX_LANGUAGES = {
        1: [],
        2: [
            "de",
            "tr",
            "fa",
            "sv-SE",
            "mn",
            "zh-CN",
            "cy",
            "ca",
            "sl",
            "et",
            "id",
            "ar",
            "ta",
            "lv",
            "ja",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
        version: int = 2,
    ) -> None:
        assert version in self.VERSIONS and split in self.SPLITS
        assert source_language is not None
        self.no_translation = target_language is None
        if not self.no_translation:
            assert "en" in {source_language, target_language}
            if source_language == "en":
                assert target_language in self.EN_XX_LANGUAGES[version]
            else:
                assert source_language in self.XX_EN_LANGUAGES[version]
        else:
            # Hack here so that we can get "split" column from CoVoST TSV.
            # Note that we use CoVoST train split for ASR which is an extension
            # to Common Voice train split.
            target_language = "de" if source_language == "en" else "en"

        self.root: Path = Path(root)

        cv_tsv_path = self.root / "validated.tsv"
        assert cv_tsv_path.is_file()

        covost_url = self.COVOST_URL_TEMPLATE.format(
            src_lang=source_language, tgt_lang=target_language
        )
        covost_archive = self.root / Path(covost_url).name
        if not covost_archive.is_file():
            download_url(covost_url, self.root.as_posix(), hash_value=None)
        extract_archive(covost_archive.as_posix())

        cv_tsv = load_df_from_tsv(cv_tsv_path)
        covost_tsv = load_df_from_tsv(
            self.root / Path(covost_url).name.replace(".tar.gz", "")
        )
        df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=covost_tsv[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        if split == "train":
            df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
        else:
            df = df[df["split"] == split]
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.root / "clips" / e["path"]
                if not path.exists() and path.with_suffix('.wav').exists():
                    path = path.with_suffix('.wav')
                    e["path"] = path.name
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        translation = None if self.no_translation else data["translation"]
        speaker_id = data["client_id"]
        _id = Path(data["path"]).with_suffix('').as_posix()
        return waveform, sample_rate, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute() / args.src_lang
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    if not args.use_audio_input:
        # Extract features
        feature_root = root / "fbank80"
        feature_root.mkdir(exist_ok=True)
        for split in CoVoST.SPLITS:
            print(f"Fetching split {split}...")
            dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
            print("Extracting log mel filter bank features...")
            for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                extract_fbank_features(
                    waveform, sample_rate, feature_root / f"{utt_id}.npy"
                )
        # Pack features into ZIP
        zip_path = root / "fbank80.zip"
        print("ZIPing features...")
        create_zip(feature_root, zip_path)
        print("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(zip_path)
        # Clean up
        shutil.rmtree(feature_root)

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    task = f"asr_{args.src_lang}"
    if args.tgt_lang is not None:
        task = f"st_{args.src_lang}_{args.tgt_lang}"
    for split in CoVoST.SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        if args.prepend_tgt_lang_tag:
            manifest["tgt_lang"] = []
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
        for i, (wav, sr, src_utt, tgt_utt, speaker_id, utt_id) in enumerate(tqdm(dataset)):
            manifest["id"].append(utt_id)
            duration_ms = int(wav.size(1) / sr * 1000)
            manifest["duration_ms"].append(duration_ms)
            if args.use_audio_input:
                audiofile = root / "clips" / dataset.data[i]["path"]
                assert audiofile.suffix == '.wav', \
                    f"You must convert the audio files to WAV format " \
                    f"(and resampling to 16 kHz is recommended):\n\n" \
                    f"{WAV_MESSAGE}\n"
                manifest["audio"].append(audiofile)
                manifest["n_frames"].append(wav.size(1))
            else:
                manifest["audio"].append(zip_manifest[utt_id])
                manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
            manifest["tgt_text"].append(src_utt if args.tgt_lang is None else tgt_utt)
            manifest["speaker"].append(speaker_id)
            if args.prepend_tgt_lang_tag:
                manifest["tgt_lang"].append(
                    args.tgt_lang if args.tgt_lang else args.src_lang
                )
        is_train_split = split.startswith("train")
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, root / f"{split}_{task}.tsv")
    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_{task}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        special_symbols = [
            f'<lang:{args.tgt_lang if args.tgt_lang else args.src_lang}>'
        ] if args.prepend_tgt_lang_tag else None
        gen_vocab(
            Path(f.name),
            root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
            special_symbols=special_symbols,
        )
    # Generate config YAML
    gen_config_yaml(
        root,
        spm_filename_prefix + ".model",
        yaml_filename=f"config_{task}.yaml",
        prepend_tgt_lang_tag=args.prepend_tgt_lang_tag,
        specaugment_policy="lb" if not args.use_audio_input else None,
        use_audio_input=args.use_audio_input,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=1000, type=int)
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", type=str)
    parser.add_argument("--use-audio-input", action='store_true',
                        help="Use raw audio, instead of extracting features.")
    parser.add_argument("--prepend-tgt-lang-tag", action='store_true',
                        help="Prepend the target language tag when loading "
                             "target sentences.")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
