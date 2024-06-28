KITTI_CONFIG="FCGF/network_configs/test-kitti.yaml"
KITTI_MODEL_PATH="KITTI-v0.3-ResUNetBN2C-conv1-5-nout32.pth"

EXPLR_CONFIG="FCGF/network_configs/test-ExpLR.yaml"
EXPLR_MODEL_PATH="20230818-ExpLR-train/20230818-15-49-41/20230821-16-32-25/20230824-13-23-48/20230825-17-59-02/20230905-12-28-32/checkpoint_100.pth"

CYCLE_POINT4_CONFIG="FCGF/network_configs/test-OneCycle-3.0-0.4.yaml"
CYCLE_POINT4_MODEL_PATH="20230821-OneCycle-LR.01-posthresh.4-negthresh-3.0/20230821-17-26-45/20230824-13-15-10/20230825-18-13-07/20230905-12-19-21/checkpoint_96.pth"

CYCLE_POINT1_CONFIG="FCGF/network_configs/test-OneCycle-1.4-0.1.yaml"
CYCLE_POINT1_MODEL_PATH="20230821-OneCycle-LR.01-posthresh.1-negthresh-1.4/20230821-17-27-51/20230824-13-09-45/20230825-18-08-17/20230905-12-23-52/checkpoint_96.pth"

NOISE="crop"
OVERLAP="0.2"
MODEL="cycle_point4"
COMPUTE=false
EVAL=true

if [[ $MODEL == "expLR" ]]; then
    NETWORK_CONFIG=$EXPLR_CONFIG
    MODEL_PATH=$EXPLR_MODEL_PATH
elif [[ $MODEL == "cycle_point4" ]]; then
    NETWORK_CONFIG=$CYCLE_POINT4_CONFIG
    MODEL_PATH=$CYCLE_POINT4_MODEL_PATH
elif [[ $MODEL == "cycle_point1" ]]; then
    NETWORK_CONFIG=$CYCLE_POINT1_CONFIG
    MODEL_PATH=$CYCLE_POINT1_MODEL_PATH
elif [[ $MODEL == "kitti" ]]; then
    NETWORK_CONFIG=$KITTI_CONFIG
    MODEL_PATH=$KITTI_MODEL_PATH
else
    echo "Unknown model $MODEL"
    exit 1
fi

CONFIG_FOLDER="FCGF/mbes_data/configs/tests/meters"
MBES_CONFIG="$CONFIG_FOLDER/$NOISE/mbesdata_${NOISE}_meters_pairoverlap=$OVERLAP.yaml"
RESULTS_ROOT="20230711-$NOISE-meters-pairoverlap=$OVERLAP/${MODEL_PATH}"
mkdir -p $RESULTS_ROOT

logname="$RESULTS_ROOT/mbes_test-$NOISE-$OVERLAP-$(basename $NETWORK_CONFIG .yaml).log"
echo $(date "+%Y%m%d-%H-%M-%S")
echo "======================================="

if [[ $COMPUTE == true ]]; then
    echo "Running mbes_test.py on noise=$NOISE overlap=$OVERLAP, network=$NETWORK_CONFIG..."
    echo "Using mbes_config=$MBES_CONFIG..."
    echo "logging to $logname..."

    python FCGF/mbes_test.py \
        --mbes_config  $MBES_CONFIG\
        --network_config $NETWORK_CONFIG \
        | tee $logname
fi

if [[ $EVAL == true ]]; then
    echo "======================================="
    echo "Evaluating results at $RESULTS_ROOT..."
    python mbes-registration-data/src/evaluate_results.py \
        --results_root $RESULTS_ROOT \
        --use_transforms pred \
        | tee $RESULTS_ROOT/eval-res-$NOISE-$OVERLAP.log
fi
echo "Done!"
echo $(date "+%Y%m%d-%H-%M-%S")
echo "======================================="