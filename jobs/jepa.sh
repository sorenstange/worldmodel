#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"

### -- set the job Name --
#BSUB -J train_jepa

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

### request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"

#BSUB -u s204229@student.dtu.dk

### -- send notification at start --
#BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o /zhome/d3/0/155487/worldmodel/outputs/jepa/%J.out
#BSUB -e /zhome/d3/0/155487/worldmodel/outputs/jepa/%J.err

# -- end of LSF options --

cd /zhome/d3/0/155487/worldmodel
uv run src/jepa.py