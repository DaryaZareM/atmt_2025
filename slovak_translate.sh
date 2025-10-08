#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2
#SBATCH --output=slovak_translate.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

python translate.py \
    --cuda \
    --input slovak_data/test.sk \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output slovak_output.en \
    --max-len 300 \
    --bleu \
    --reference slovak_data/test.en
