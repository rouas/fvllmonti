#!/usr/bin/env python3

import torch
import os

# new addidtion 
import numpy as np
import pandas as pd

import sys, getopt

def main(argv):
        inputfile=''
        espnet2 = False  # Default value for espnet2 option
        try:
                opts, args = getopt.getopt(argv, "hi:2", ["ifile=", "espnet2"])
        except getopt.GetoptError:
                print ('test.py -i <inputfile> [-2]')
                sys.exit(2)
        for opt, arg in opts:
                if opt == '-h':
                        print ('test.py -i <inputfile> [-2]')
                        sys.exit()
                elif opt in ("-i", "--ifile"):
                        inputfile = arg
                elif opt == '-2':
                        espnet2 = True
        print ('Input file is', inputfile)
        print ('ESPnet2 option is', espnet2)


        if espnet2:
                print("using espnet2 load method")
                from pathlib import Path
                from espnet2.tasks.asr import ASRTask
                config_file = Path(inputfile).parent / "config.yaml"
                task = ASRTask
                m, train_args = task.build_model_from_file(config_file, inputfile, 'cpu')
        else:
                                
                #        m = torch.load("saved_models/pruned_tiles_quantized/The_model_complet")
                #        m = torch.load(inputfile)
                from espnet.asr.pytorch_backend.asr import load_trained_model
                m, train_args = load_trained_model(inputfile, training=False)
        

        torch.set_printoptions(profile="full") # Pour afficher toutes les valeurs

        GLOBNZ=TOTAL=0
        for name, module in m.named_modules():
                #if "coders." in name:
                #if "feed" in name:
                if isinstance(module, torch.nn.Linear):
                        if not "ctc" in name and not "embed" in name:
                                
#                                print(name, "module name:   ", module)
                                #print("*******************")
                                
                                
                                # let's try to see if we can compute a sparsity:
                                nzeros=ntotal=0
                                nzeros += float(torch.sum(module.weight == 0))
                                ntotal += float(module.weight.nelement())
                                GLOBNZ+=nzeros
                                TOTAL+=ntotal
                                print(name, "module name:   ", module, "sparsity : ", 100 * nzeros/ntotal, "(",nzeros,"/",ntotal,")")

        print("global pruning rate = ",GLOBNZ/TOTAL, "(",GLOBNZ,"/",TOTAL,")")
                        
if __name__ == "__main__":
        main(sys.argv[1:])

                
                
