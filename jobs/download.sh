#!/bin/sh

### General options
### –- specify queue --
#BSUB -q hpc

### -- set the job Name --
#BSUB -J download

### -- ask for number of cores (default: 1) --
#BSUB -n 1

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

#BSUB -o /zhome/d3/0/155487/worldmodel/outputs/download_%J.out
#BSUB -e /zhome/d3/0/155487/worldmodel/outputs/download_%J.err

# -- end of LSF options --

cd /zhome/d3/0/155487/worldmodel
uv run src/data.py
