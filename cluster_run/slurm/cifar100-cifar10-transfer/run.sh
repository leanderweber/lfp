mkdir -p /mnt/output
mkdir -p /mnt/input
cd /mnt/layer-wise-feedback-propagation

echo "STARTING JOB $@"

python3 -m run_resubmission1_experiment --config_file "configs/cifar100-cifar10-transfer/cluster/$@"

