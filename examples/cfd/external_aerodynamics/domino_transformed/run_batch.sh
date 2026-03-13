#!/bin/bash
#SBATCH --exclusive
#SBATCH -p batch
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --job-name=domino-transformed
#SBATCH --time=03:30:00
#SBATCH --account=coreai_modulus_cae
#SBATCH --dependency=singleton
#SBATCH -o ./sbatch_logs/%x_%j.out
#SBATCH -e ./sbatch_logs/%x_%j.err

#--mail-type=FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90,END
#set -euxo pipefail

# Mount relevant folders and image to lustre/
readonly _cont_mounts="/lustre/:/lustre/"
readonly _cont_image='/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/snidhan/physicsnemo25.11.sqsh'

export NPROC_PER_NODE=4
export TOTAL_GPU=$(($SLURM_JOB_NUM_NODES * $NPROC_PER_NODE))

# train the model
RUN_CMD="python -u train.py"

echo "Running on hosts: $(echo $(scontrol show hostname))"
srun -A coreai_modulus_cae  \
     --container-image="${_cont_image}" \
     --container-mounts="${_cont_mounts}" \
     --ntasks-per-node=4 \
     bash -c "
     ldconfig
     set -x
     export CUDNN_V8_API_ENABLED=1
     export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
     unset TORCH_DISTRIBUTED_DEBUG
     cd /lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/rranade/physicsnemo-fork/rishi-fork/modulus/examples/cfd/external_aerodynamics/domino_transformed
     source myenv/bin/activate
     export LD_LIBRARY_PATH=/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/rranade/physicsnemo-fork/rishi-fork/modulus/examples/cfd/external_aerodynamics/domino_transformed/myenv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
     ${RUN_CMD}"
