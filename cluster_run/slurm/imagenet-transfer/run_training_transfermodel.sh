for config_file in ../../../configs/imagenet-transfer/cluster/*.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer_transfermodel.sh $config_file
    sleep 0.5
done;
