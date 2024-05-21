#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=15G
#SBATCH --job-name=ABAW6_preprocess
#SBATCH --time=15-00:00:00
#SBATCH --mail-user=gnana-praveen.rajasekhar@crim.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
export PATH="/misc/scratch11/anaconda3/bin:$PATH"
#source activate py36_ssd20
source activate pre
#conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
#pip install matplotlib
#pip3 install -U scikit-learn scipy matplotlib
#pip3 install opencv-python
#pip3 install tqdm
#pip3 install pandas
#pip3 install resampy
#pip3 install soundfile
#pip3 install tensorboard
#conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
scripts_dir=`pwd`
MatlabFE=`pwd`
mdl=senet18e17
lf=ocsoftmax
atype=LA

#ptd=/misc/archive07/patx/asvspoof2019
#ptf=/lu/fast_scratch/patx/alamja/anti-spoofing-10dec2020/${atype}pickle
#ptp=/misc/archive07/patx/asvspoof2019/${atype}/ASVspoof2019_${atype}_protocols
#mdl_dir=/lu/fast_scratch/patx/alamja/anti-spoofing-10dec2020/models/$atype/$mdl/$lf
##mdl_dir=/lu/fast_scratch/patx/alamja/anti-spoofing-10dec2020/models_ensem/$atype/$mdl/$lf
#mkdir -p ${mdl_dir}/
#echo "This is array task ${SLURM_ARRAY_TASK_ID}, the sample name is ${data_start_range} and the sex is ${data_end_range}." >> output.txt
#python train_senet18e17.py --add_loss ${lf} -o ${mdl_dir} --gpu 0 -a ${atype} -d ${ptd} -f ${ptf} -p ${ptp}
#export CUDA_VISIBLE_DEVICES=0
python3 main.py
wait
echo "DONE"