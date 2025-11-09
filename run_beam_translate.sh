#!/usr/bin/bash -l
#SBATCH --job-name=beam_translate
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=2
#SBATCH --output=out_assignment2.out
#SBATCH --error=err_assignment2.err

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit


# TRANSLATE
python translate.py \
    --cuda \
    --input cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path model/checkpoints/checkpoint_best.pt \
    --output cz-en/output1.txt \
    --max-len 300\
    --beam-size 5