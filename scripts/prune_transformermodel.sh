#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -m <model_file> [OPTIONS]"
    echo "Options:"
    echo "  -a, --prune-asr-model"
    echo "  -b, --asr-model-stats"
    echo "  -g, --tile"
    echo "  -t, --thres <value>"
    echo "  -i, --prune-asr-model-local"
    echo "  -q, --prune-asr-model-tile-percentV2"
    echo "  -s, --save-to <file>"
    echo "  -h, --help"
    echo "  -v, --verbose"
    echo "  -2, --espnet2"
    echo "  -3, --mttask"
    echo "  -T, --task <ASRTask>"
    exit 1
}

# Initialize variables with default values
verbose=0
prune_asr_model=false
asr_model_stats=false
prune_asr_model_tile_percentV2=false
tile=2
thres=0.6
save_to=""
model=""
espnet2=false
mttask=false
task="ASRTask"

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -v|--verbose)
            verbose=$2
	    shift
            ;;
        -a|--prune-asr-model)
            prune_asr_model=true
            ;;
        -b|--asr-model-stats)
            asr_model_stats=true
            ;;
        -g|--tile)
            tile=$2
	    shift
            ;;
        -t|--thres)
            thres="$2"
            shift
            ;;
        -q|--prune-asr-model-tile-percentV2)
            prune_asr_model_tile_percentV2=true
            ;;
        -s|--save-to)
            save_to="$2"
            shift
            ;;
        -m|--model)
            model="$2"
            shift
            ;;
        -2|--espnet2)
            espnet2=true
            ;;
	    -3|--mttask)
            mttask=true
            ;;
        -T|--task)
            task="$2"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $key"
            usage
            ;;
    esac
    shift
done

# Validate model option
if [[ -z $model ]]; then
    echo "Error: Please specify a model with -m or --model"
    usage
fi

# Validate options
if [[ -z $save_to ]]; then
    echo "Error: Missing --save-to option"
    usage
fi

# Example usage of options

if [[ $prune_asr_model_tile_percentV2 == true ]]; then
    echo "Prune ASR Model tile bc (block pruning) option selected"
    echo "block size = $tile"
    echo "threshold = $thres"
fi

if [[ $save_to ]]; then
    echo "Saving output to: $save_to"
fi

if [[ $espnet2 == true ]]; then
    echo "assuming espnet2 model"
else
    echo "assuming espnet1 model"
fi

echo "mttask = $mttask"

if [[ $mttask == true ]]; then
    echo "using MT TASK for the espnet2 blabla"
else
    echo "not using MT Task -> ASRTask"
fi

echo "task = $task"
if [[ $task == "ASRTask" ]]; then
    echo "using ASRTask"
else
    if [[ $task == "MTTask" ]]; then
        echo "using MTTask"
    else
        if [[ $task == "STTask" ]]; then
            echo "using STTask"
        else
            echo "task $task not recognized"
        fi
    fi
fi



echo "Model specified: $model"


#. ~/Sources/git/espnet/tools/activate_python.sh
. ~/Sources/git/espnet2025/espnet/tools/activate_python.sh

prunemodel.py \
    --prune-asr-model $prune_asr_model \
    --asr-model-stats $asr_model_stats \
    --prune-asr-model-tile-percentV2 $prune_asr_model_tile_percentV2 \
    --tile $tile \
    --thres $thres \
    --model  ${model} \
    --espnet2 $espnet2 \
    --mttask $mttask \
    --task $task \
    --save-to $save_to \
    --verbose $verbose

if [[ $espnet2 == true ]]; then
    echo "saving as espnet2 packed model?"
    packed_model="${save_to}.espnet2.zip"
    config_file=$(dirname $model)/config.yaml
    python3 -m espnet2.bin.pack asr \
        --asr_train_config $config_file \
        --asr_model_file $save_to \
        --outpath "${packed_model}"
fi



date
