#!/bin/bash
#SBATCH --job-name=moe_verify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --partition=gpu_requeue
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err

mkdir -p logs

module load python
module load cuda
conda activate torchtitan

# Force IPv4 -- holygpu nodes resolve to IPv6 by default, which c10d doesn't support
export MASTER_ADDR=$(getent ahostsv4 "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)" | awk 'NR==1{print $1}')
export MASTER_PORT=29500
export PYTHONPATH=$HOME/torchtitan:$PYTHONPATH

cd $HOME/torchtitan

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m torchtitan.train \
  --model.name deepseek_v3 \
  --model.config deepseek_v3_debugmodel \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.expert_parallel_degree 2 \
  --parallelism.context_parallel_degree 1 \
  --parallelism.data_parallel_shard_degree 1 \
  --training.steps 3
