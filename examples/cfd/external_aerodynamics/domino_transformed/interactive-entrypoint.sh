#!/bin/bash

readonly _cont_mounts="/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/:/lustre/"
readonly _cont_image="/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/snidhan/physicsnemo25.11.sqsh"

srun -A coreai_modulus_cae \
    -p batch \
    --nodes=1 \
    --gpus 4 \
    --container-image=${_cont_image} \
    --container-mounts=${_cont_mounts} \
    --job-name=coreai_modulus_cae-modulus:shift-benchmarks \
    -t 03:30:00 \
    --pty bash
