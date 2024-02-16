#!/usr/bin/env python3

import torch
import os

# new addidtion 
import numpy as np
import pandas as pd

import sys, getopt

def main(argv):
        inputfile = ''
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
        for opt, arg in opts:
                if opt == '-h':
                        print ('test.py -i <inputfile>')
                        sys.exit()
                elif opt in ("-i", "--ifile"):
                        inputfile = arg
        print ('Input file is ', inputfile)


                        
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

                
                
