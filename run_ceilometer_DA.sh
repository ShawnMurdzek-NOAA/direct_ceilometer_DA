#!/bin/bash
#SBATCH -M c6
#SBATCH -A bil-pmp
#SBATCH -J run_ceilometer_DA
#SBATCH -o %x-%j.out
#SBATCH -t 1:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

code_path=/ncrc/home2/Shawn.S.Murdzek/src/direct_ceilometer_DA
yml_fname=sample.yml

date

# Copy over ceilometer DA code
if [ -d direct_ceilometer_DA ]; then
  rm -rf ./direct_ceilometer_DA
fi
cp -r ${code_path} .

# Activate Python environment
module load python
source activate my_py
#export PYTHONPATH=$PYTHONPATH:/ncrc/home2/Shawn.S.Murdzek/src/
which python

# Run code
cd direct_ceilometer_DA
python -u ceilometer_obs_enkf.py ../${yml_fname}

date
