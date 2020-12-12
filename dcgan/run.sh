#!/bin/sh
#PBS -q h-short
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1
#PBS -l walltime=1:00:00
#PBS -W group_list=gp14
cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh
source /lustre/gp14/p14009/.bashrc
conda activate /lustre/gp14/p14009/anaconda3/envs/generate_airfoil
module purge
module load anaconda3/2020.07 cuda10/10.2.89
module load pytorch/1.5.0
cd generate_airfoil/dcgan
python train.py
