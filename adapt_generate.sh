#!/bin/bash

export ROOT=.
export FAIRSEQ_ROOT=.
export WAV2VEC_ROOT=wav2vec/pre-trained-ckp
export MBART_ROOT=mbart/pre-trained-ckp 
export MUSTC_ROOT=data/mustc
export HYDRA_FULL_ERROR=1

MODEL_PATH=/path/to/model
fairseq-generate "${DATA_ROOT}" \
  --path $MODEL_PATH \
  --user-dir ${ROOT}/fairseq_modules \
  --task speech_to_text_iwslt21 --gen-subset tst-COMMON_mustc \
  --max-source-positions 960000 --max-tokens 960000   \
  --skip-invalid-size-inputs-valid-test --prefix-size 1 \
  --beam 5 --scoring sacrebleu


