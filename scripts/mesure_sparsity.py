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
        mttask = False # false = use ASR task for loading model
        try:
                opts, args = getopt.getopt(argv, "hi:23", ["ifile=", "espnet2","mttask"])
        except getopt.GetoptError:
                print ('mesure_sparsity.py -i <inputfile> [-2] [-3]')
                sys.exit(2)
        for opt, arg in opts:
                if opt == '-h':
                        print ('test.py -i <inputfile> [-2]')
                        sys.exit()
                elif opt in ("-i", "--ifile"):
                        inputfile = arg
                elif opt == '-2':
                        espnet2 = True
                elif opt == '-3':
                        mttask = True
        print ('Input file is', inputfile)
        print ('ESPnet2 option is', espnet2)
        print ('MTtask option is', mttask)


        if espnet2:
                print("using espnet2 load method")
                from pathlib import Path
                
                config_file = Path(inputfile).parent / "config.yaml"
                if mttask:
                        print("using MTTask")
                        from espnet2.tasks.mt import MTTask
                        task = MTTask
                else:
                        print("using ASRTask")
                        from espnet2.tasks.asr import ASRTask
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
                                
                                # mesure la moyenne pour chaque block
                                moy=float(torch.mean(torch.abs(module.weight)))
                                
                                # let's try to see if we can compute a sparsity:
                                nzeros=ntotal=0
                                nzeros += float(torch.sum(module.weight == 0))
                                ntotal += float(module.weight.nelement())
                                GLOBNZ+=nzeros
                                TOTAL+=ntotal
                                print(name, "module name:   ", module, "sparsity : ", 100 * nzeros/ntotal, "(",nzeros,"/",ntotal,")","moyenne",moy)

        print("global pruning rate = ",GLOBNZ/TOTAL, "(",GLOBNZ,"/",TOTAL,")")
                        
if __name__ == "__main__":
        main(sys.argv[1:])

                
                
