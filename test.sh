configs_folder="mbes_data/configs/tests/meters"
log_folder=""
network_configs=$(ls "network_configs")

for folder in $(ls $configs_folder)
do
    subfolder="$configs_folder/$folder"
    configs=$(ls $subfolder)
    for c in $configs
    do
        mbes_config="$subfolder/$c"
        for n in $network_configs
        do
            network_config="network_configs/$n"
            logname="$log_folder/$(basename $c .yaml)_$(basename $n .yaml).log"
            echo "======================================="
            echo "Using $mbes_config..."
            echo "Using $network_config..."
            echo "Logging to $logname..."
            python mbes_test.py --mbes_config $mbes_config --network_config $network_config > $logname
        done
    done
done
