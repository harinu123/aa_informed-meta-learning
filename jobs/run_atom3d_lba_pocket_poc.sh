#!/usr/bin/env bash
set -euo pipefail

# Replace X_DIM with the value printed by data_gen_atom3d_lba_pocket_poc.py --dry-run
X_DIM=<X_DIM_FROM_META>

# NP
python config.py \
  --project-name INPs_atom3d_lba_poc \
  --dataset atom3d-lba-pocket-poc \
  --input-dim "${X_DIM}" --output-dim 1 \
  --run-name-prefix np_atom3d \
  --use-knowledge False \
  --min-num-context 0 --max-num-context 10 \
  --batch-size 16 --num-epochs 2000 \
  --hidden-dim 128 --x-transf-dim 128 \
  --seed 0
python models/train.py

# INP
python config.py \
  --project-name INPs_atom3d_lba_poc \
  --dataset atom3d-lba-pocket-poc \
  --input-dim "${X_DIM}" --output-dim 1 \
  --run-name-prefix inp_atom3d \
  --use-knowledge True \
  --text-encoder none --knowledge-merge sum \
  --min-num-context 0 --max-num-context 10 \
  --batch-size 16 --num-epochs 2000 \
  --hidden-dim 128 --x-transf-dim 128 \
  --knowledge-extractor-num-hidden 1 \
  --seed 1
python models/train.py

# Contrastive INP
python config.py \
  --project-name INPs_atom3d_lba_poc \
  --dataset atom3d-lba-pocket-poc \
  --input-dim "${X_DIM}" --output-dim 1 \
  --run-name-prefix clinp_atom3d \
  --use-knowledge True \
  --text-encoder none --knowledge-merge sum \
  --knowledge-contrastive True \
  --kcon-inv-weight 1.0 --kcon-use-weight 1.0 --kcon-margin 1.0 \
  --min-num-context 0 --max-num-context 10 \
  --batch-size 16 --num-epochs 2000 \
  --hidden-dim 128 --x-transf-dim 128 \
  --knowledge-extractor-num-hidden 1 \
  --seed 2
python models/train.py
