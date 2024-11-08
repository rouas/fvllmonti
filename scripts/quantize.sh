#!/bin/bash


. ~/Sources/git/espnet-vanilla/espnet/tools/activate_python.sh

quantize_model.py -i $1
