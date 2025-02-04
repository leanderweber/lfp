for config_file in ../../../configs/cifar-basemodels/cluster/*_basemodel.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer_basemodel.sh $config_file
    sleep 0.5
done;
