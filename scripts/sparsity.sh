
# Default values for options
espnet2=""

# Parse options
while getopts ":i:2" opt; do
  case $opt in
    i)
      input_file=$OPTARG
      ;;
    2)
      espnet2="-2"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

# Activate Python environment
. ~/Sources/git/espnet/tools/activate_python.sh

# Run the script with options
mesure_sparsity.py -i "$input_file" "$espnet2"