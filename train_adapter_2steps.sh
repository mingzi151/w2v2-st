#!/bin/bash

export ROOT=.
export FAIRSEQ_ROOT=.
export WAV2VEC_ROOT=pre-trained-ckp/wav2vec
export MBART_ROOT=pre-trained-ckp/mbart 
export MUSTC_ROOT=data/mustc
export SAVE_DIR=outputs
export HYDRA_FULL_ERROR=1
step=$1

export STEP=$step
export DATA_ROOT=data/mustc_only
mkdir -p $DATA_ROOT


if [ ! -f "$PWD/$DATA_ROOT/config.yaml" ];then
  echo "linking files to $DATA_ROOT"
  ln -s "$PWD"/$MUSTC_ROOT/en-de/config_st.yaml "$PWD"/$DATA_ROOT/config.yaml
  ln -s "$PWD"/$MUSTC_ROOT/en-de/spm_bpe250000_st.{txt,model} "$PWD"/$DATA_ROOT/

  ln -s "$PWD"/$MUSTC_ROOT/en-de/train_st_filtered.tsv "$PWD"/$DATA_ROOT/train_mustc.tsv
  ln -s "$PWD"/$MUSTC_ROOT/en-de/dev_st.tsv "$PWD"/$DATA_ROOT/dev_mustc.tsv
  ln -s "$PWD"/$MUSTC_ROOT/en-de/tst-COMMON_st.tsv "$PWD"/$DATA_ROOT/tst-COMMON_mustc.tsv
  ln -s "$PWD"/$MUSTC_ROOT/en-de/tst-HE_st.tsv "$PWD"/$DATA_ROOT/tst-HE_mustc.tsv
fi

if [ "$step" = "step1" ];then
  echo "training the first step"
  fairseq-hydra-train \
    --config-dir ${ROOT}/config/ \
    --config-name mustc_only_madapt_2step_1.yaml

elif [ $step = "step2" ];then
  echo "training the second step...."
  fairseq-hydra-train \
    --config-dir ${ROOT}/config/ \
    --config-name mustc_only_madapt_2step_2_block_mod.yaml

fi

