#!/bin/bash
#SBATCH --job-name=hyperparam_tuning
#SBATCH --account=small
#SBATCH --partition=small
#SBATCH --output=output/tuning_results_v2/logs/tuning_%j.out
#SBATCH --error=output/tuning_results_v2/logs/tuning_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# check available cpus with sinfo -N --Format=NodeList:20,CPUsState:20,Gres:30,GresUsed:30,StateLong:10
#SBATCH --cpus-per-task=32
# check available memory with scontrol show node 1850-node04 (or other node)
#SBATCH --mem=192G
#SBATCH --gres=gpu:1g.10gb:1
##SBATCH --nodelist=1850-node03

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lfigu067@uottawa.ca,jrost029@uottawa.ca

echo "================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "================================================"

source /home/uottawa.o.univ/$USER/venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo -e "\n=== Python Environment ==="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

python -m tuning_v2.main_tuning --model classifier --sbatch True

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\nHyperparameter tuning completed successfully!"
else
    echo -e "\nHyperparameter tuning failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo -e "\n================================================"
echo "Job finished at $(date)"
echo "Total runtime: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m $((SECONDS % 60))s"
echo "================================================"
