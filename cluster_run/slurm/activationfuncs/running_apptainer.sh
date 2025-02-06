#!/bin/bash
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=""
#SBATCH --job-name=lfp-transfer
#SBATCH --output=lfp-transfer-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

TOKEN='<git access token>'
USER_URL="<git user url>"
REPO_NAME="<repo name>"
BRANCH_NAME="<branch name>"

DATA_SOURCE_DIR="<directory where cifar datasets are stored>"
RESULT_STORAGE_DIR="<directory where results should be stored>"

mkdir -p $RESULT_STORAGE_DIR

source "/etc/slurm/local_job_dir.sh"

bash ../download_repository.sh -u $USER_URL -r $REPO_NAME -b $BRANCH_NAME -t $TOKEN -p ${LOCAL_JOB_DIR}
echo "LOCAL DIR: ${LOCAL_JOB_DIR}"

fname_config=$(basename $@)

mkdir -p ${LOCAL_JOB_DIR}/data

echo "Copying Data..."
echo ${DATA_SOURCE_DIR}
cp -r ${DATA_SOURCE_DIR}/cifar10 ${LOCAL_JOB_DIR}/data
cp -r ${DATA_SOURCE_DIR}/cifar100 ${LOCAL_JOB_DIR}/data

echo $fname_config

echo "Start Training"
apptainer run --nv \
              --bind ${LOCAL_JOB_DIR}:/mnt \
              --bind <isic dataset path>:/mnt/data/isic \
              --bind <cub dataset path>:/mnt/data/cub \
              --bind <food11 dataset path>:/mnt/data/food11 \
              --bind <imagenet dataset path>:/mnt/data/imagenet \
              ../../singularity/image_mini.sif bash /mnt/layer-wise-feedback-propagation/cluster_run/slurm/activationfuncs/run.sh $fname_config
echo "Training finished"

echo "Copying results"
cd ${LOCAL_JOB_DIR}
tar -czf ${fname_config}-output_data.tgz output
cp -r ${fname_config}-output_data.tgz ${RESULT_STORAGE_DIR}
mv ${SLURM_SUBMIT_DIR}/lfp-transfer-${SLURM_JOB_ID}.out ${RESULT_STORAGE_DIR}/lfp-transfer-${fname_config}-${SLURM_JOB_ID}.out
echo "Results Copied"
echo "Done!"
