#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -m <model_file> [OPTIONS]"
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
    echo "  -o, --prune-asr-model-tile-percent"
    echo "  -q, --prune-asr-model-tile-percentV2"
    echo "  -p, --prune-asr-model-tile-round"
    echo "  -t, --prune-asr-model-tile-att"
    echo "  -s, --save-to <file>"
    echo "  -h, --help"
    echo "  -v, --verbose"
    echo "  -2, --espnet2"
    echo "  -3, --mttask"
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
prune_asr_model_tile_percent=false
prune_asr_model_tile_percentV2=false
prune_asr_model_tile_round=false
prune_asr_model_tile_att=false
tile=2
thres=0.6
prune_asr_model_adapt=false
enc=0.5
dec=0.3
fixe=0.3
save_to=""
model=""
espnet2=false
mttask=false

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
        -o|--prune-asr-model-tile-percent)
            prune_asr_model_tile_percent=true
            ;;
        -q|--prune-asr-model-tile-percentV2)
            prune_asr_model_tile_percentV2=true
            ;;
        -p|--prune-asr-model-tile-round)
            prune_asr_model_tile_round=true
            ;;
        -t|--prune-asr-model-tile-att)
            prune_asr_model_tile_att=true
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

if [[ $espnet2 == true ]]; then
    echo "assuming espnet2 model"
else
    echo "assuming espnet1 model"
fi

echo $mttask

if [[ $mttask == true ]]; then
    echo "using MT TASK for the espnet2 blabla"
else
    echo "not using MT Task -> ASRTask"
fi


echo "Model specified: $model"


#. ~/Sources/git/espnet/tools/activate_python.sh
. ~/Sources/git/espnet-vanilla/espnet/tools/activate_python.sh

prunemodel.py \
    --prune-asr-model $prune_asr_model \
    --asr-model-stats $asr_model_stats \
    --prune-asr-model-local $prune_asr_model_local \
    --prune-asr-model-adapt $prune_asr_model_adapt \
    --prune-asr-model-tile-bc $prune_asr_model_tile_bc \
    --prune-asr-model-tile-percent $prune_asr_model_tile_percent \
    --prune-asr-model-tile-percentV2 $prune_asr_model_tile_percentV2 \
    --prune-asr-model-tile-att $prune_asr_model_tile_att \
    --prune-asr-model-tile-round $prune_asr_model_tile_round \
    --am ${am} \
    --amAtt ${amAtt} \
    --enc $enc \
    --dec $dec \
    --fixe ${fixe} \
    --tile $tile \
    --thres $thres \
    --tileFF true \
    --model  ${model} \
    --espnet2 $espnet2 \
    --mttask $mttask \
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
