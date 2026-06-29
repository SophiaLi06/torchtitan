#!/bin/bash
#SBATCH --job-name=moe_verify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # ONE srun task per node; torchrun spawns the GPU processes
#SBATCH --gpus=nvidia_a100-sxm4-80gb:4
#SBATCH --cpus-per-task=16        # give torchrun enough CPUs to spawn 4 workers
#SBATCH --mem=64G                 # 4 CUDA processes * ~2 GB CPU RAM each + dataset/tokenizer overhead
#SBATCH --time=00:15:00
#SBATCH --partition=seas_gpu
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err

mkdir -p logs

module load python
module load cuda
conda activate torchtitan

export PYTHONPATH=$HOME/torchtitan:$PYTHONPATH

cd $HOME/torchtitan

# Single-node: use localhost so no cross-node networking is needed
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29500 \
  -m torchtitan.train \
  --module deepseek_v3 \
  --config deepseek_v3_debugmodel \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.expert_parallel_degree 2 \
  --parallelism.context_parallel_degree 1 \
  --parallelism.data_parallel_shard_degree 1 \
  --training.steps 3
