#!/bin/bash
#SBATCH -J finetune                      # 作业名称，建议改得更有区分度
#SBATCH -p normal                        # 使用 normal 分区（生产任务推荐）
#SBATCH -A mscitsuperpod                 # 你的账号（已确认）
#SBATCH -N 1                             # 使用 1 个节点
#SBATCH --gpus-per-node=1                # 推荐写法：每节点申请 1 张 GPU
#SBATCH --cpus-per-task=8                # 分配 8 个 CPU 核心（配合 vLLM 使用）
#SBATCH -t 01:30:00                      # 时间限制：1小时30分（根据需要调整）
#SBATCH --mem=64G                        # 内存

#SBATCH --mail-user=txueae@connect.ust.hk
#SBATCH --mail-type=BEGIN,END,FAIL

# ==================== 环境设置 ====================
export HF_HOME=/tmp/${USER}/huggingface_cache
mkdir -p ${HF_HOME}

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU Info:"
nvidia-smi

# ==================== 运行 ====================
python run_py_finetune.sh
echo "Evaluation completed at $(date)"