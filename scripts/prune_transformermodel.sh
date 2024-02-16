#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -m <model_name> [OPTIONS]"
    echo "Options:"
    echo "  -a, --prune-asr-model"
    echo "  -b, --asr-model-stats"
    echo "  -c, --prune-asr-model-local"
    echo "  -d, --amAtt"
    echo "  -e, --am"
    echo "  -f, --prune-asr-model-tile-bc"
    echo "  -g, --tile"
    echo "  -t, --thres <value>"
    echo "  -i, --prune-asr-model-adapt"
    echo "  -j, --enc <value>"
    echo "  -k, --dec <value>"
    echo "  -l, --fixe <value>"
    echo "  -s, --save-to <file>"
    echo "  -h, --help"
    echo "  -v, --verbose"
    exit 1
}

# Initialize variables with default values
verbose=0
prune_asr_model=false
asr_model_stats=false
prune_asr_model_local=false
amAtt=0.3
am=0.3
prune_asr_model_tile_bc=false
tile=2
thres=0.6
prune_asr_model_adapt=false
enc=0.5
dec=0.3
fixe=0.3
save_to=""
model=""

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
        -c|--prune-asr-model-local)
            prune_asr_model_local=true
            ;;
        -d|--amAtt)
            amAtt=$2
	    shift
            ;;
        -e|--am)
            am=$2
	    shift
            ;;
        -f|--prune-asr-model-tile-bc)
            prune_asr_model_tile_bc=true
            ;;
        -g|--tile)
            tile=$2
	    shift
            ;;
        -t|--thres)
            thres="$2"
            shift
            ;;
        -i|--prune-asr-model-adapt)
            prune_asr_model_adapt=true
            ;;
        -j|--enc)
            enc="$2"
            shift
            ;;
        -k|--dec)
            dec="$2"
            shift
            ;;
        -l|--fixe)
            fixe="$2"
            shift
            ;;
        -s|--save-to)
            save_to="$2"
            shift
            ;;
        -m|--model)
            model="$2"
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
if [[ $prune_asr_model_local == true ]]; then
    echo "Prune ASR Model local (simple pruning) option selected"
    echo "attention rate = $amAtt"
    echo "FF rate = $am"
fi

if [[ $prune_asr_model_adapt == true ]]; then
    echo "Prune ASR Model adapt (variable rate) option selected"
    echo "attention rate = $fixe"
    echo "encoder rate = $enc"
    echo "decoder rate = $dec"
fi

if [[ $prune_asr_model_tile_bc == true ]]; then
    echo "Prune ASR Model tile bc (block pruning) option selected"
    echo "block size = $tile"
    echo "threshold = $thres"
fi

if [[ $save_to ]]; then
    echo "Saving output to: $save_to"
fi

echo "Model specified: $model"


. ~/Sources/git/espnet/tools/activate_python.sh

prunemodel.py \
    --prune-asr-model $prune_asr_model \
    --asr-model-stats $asr_model_stats \
    --prune-asr-model-local $prune_asr_model_local \
    --prune-asr-model-adapt $prune_asr_model_adapt \
    --prune-asr-model-tile-bc $prune_asr_model_tile_bc \
    --am ${am} \
    --amAtt ${amAtt} \
    --enc $enc \
    --dec $dec \
    --fixe ${fixe} \
    --tile $tile \
    --thres $thres \
    --tileFF true \
    --model  ${model} \
    --save-to $save_to \
    --verbose $verbose

date
