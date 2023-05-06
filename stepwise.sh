#!/usr/bin/env bash
##SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=zs0608@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=e_stpwz
#SBATCH --output=slurm/%x_%j.log

# why are we not doing single shot
STEPWISE=/n/fs/484-nlproofs/zs0608/NLProofS/prover/lightning_logs/version_18980624/results_test.tsv
BLEURT=/n/fs/484-nlproofs/am6723/bleurt-large-512
OUTPUTS=outputs

source /n/fs/484-nlproofs/miniconda3/bin/activate
conda activate entbank

echo "@== STEPWISE ==@"
python eval/run_scorer.py \
  --task "task_2" \
  --split test \
  --prediction_file $STEPWISE  \
  --output_dir  $OUTPUTS/real_gpt  \
  --bleurt_checkpoint $BLEURT

