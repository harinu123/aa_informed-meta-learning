#!/usr/bin/env bash
set -euo pipefail

# NP baseline
python config.py \
  --project-name INPs_modadd \
  --dataset modular-addition-rotations \
  --input-dim 2 --output-dim 2 \
  --run-name-prefix np_modadd \
  --use-knowledge False \
  --knowledge-type none \
  --text-encoder set --knowledge-merge sum \
  --min-num-context 0 --max-num-context 64 \
  --batch-size 64 \
  --num-epochs 3000 \
  --noise 0.0 \
  --mod-p 113 \
  --mod-episode-size 2048 \
  --mod-m-train-max 20 \
  --mod-m-test-min 21 --mod-m-test-max 40 \
  --seed 0
python models/train.py

# INP with weak knowledge (ω)
python config.py \
  --project-name INPs_modadd \
  --dataset modular-addition-rotations \
  --input-dim 2 --output-dim 2 \
  --run-name-prefix inp_w_modadd \
  --use-knowledge True \
  --knowledge-type w \
  --text-encoder set --knowledge-merge sum \
  --knowledge-dropout 0.0 \
  --min-num-context 0 --max-num-context 64 \
  --batch-size 64 \
  --num-epochs 3000 \
  --noise 0.0 \
  --mod-p 113 \
  --mod-episode-size 2048 \
  --mod-m-train-max 20 \
  --mod-m-test-min 21 --mod-m-test-max 40 \
  --test-num-z-samples 32 \
  --seed 0
python models/train.py
