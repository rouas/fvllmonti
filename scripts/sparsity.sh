
# Default values for options
espnet2=""
mttask=""


# Parse options
while getopts ":i:23" opt; do
  case $opt in
    i)
      input_file=$OPTARG
      ;;
    2)
      espnet2="-2"
      ;;
    3)
      mttask="-3"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Check if input_file is provided
if [ -z "$input_file" ]; then
  echo "Error: -i option (input file) is required." >&2
  exit 1
fi



# Activate Python environment
#. ~/Sources/git/espnet/tools/activate_python.sh
. ~/Sources/git/espnet-vanilla/espnet/tools/activate_python.sh
# Run the script with options
#echo mesure_sparsity.py -i "$input_file" "$espnet2" "$mttask"
mesure_sparsity.py -i "$input_file" "$espnet2" "$mttask"