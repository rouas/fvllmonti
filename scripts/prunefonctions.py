# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""PRUNING for the speech recognition task."""

import copy
import itertools
import json
import logging
import math
import os
import time

# freom asr_utils
import argparse

# from asr_init.py
import importlib

import numpy as np
import torch

#lbz
import torch.nn.utils.prune as prune
import torch.nn.functional as F

# jlr
from collections import Counter


def prunetransformer(args):
    """prune with the given args.

    Args:
        args (namespace): The program arguments.

    """
        
    model, train_args = load_trained_model(args.model, training=False)
        
    if args.verbose:
        print("pruning.py: Inference arguments...  ",args)

    print(
        "prune transformer:  initial model = number of parameters="
        + str(sum(p.numel() for p in model.parameters()))
        + " encoder= "
        + str(sum(p.numel() for p in model.encoder.parameters()))
        + " decoder= "
        + str(sum(p.numel() for p in model.decoder.parameters()))
    )
 
    if args.asr_model_stats:
        
        conv1d = conv2d = mha = ff = 0
        for name, param in model.named_parameters():
            if ".conv." in name:
                if "weight" in name:
                    if "norm" not in name:
                        print(name, param.size(), param.shape, sum(p.numel() for p in param))
                        conv2d += sum(p.numel() for p in param)
            elif ".conv_module" in name:
                if "weight" in name:
                    if "norm" not in name:
                        print(name, param.size(), param.shape, sum(p.numel() for p in param))
                        conv1d += sum(p.numel() for p in param)
                #print(name, type(param), param.size(), param.shape, param[:2])
            elif "self_attn" in name:
                if "weight" in name:
                    print(name, param.size(), param.shape, sum(p.numel() for p in param))
                    mha += sum(p.numel() for p in param)
            elif "feed_forward" in name:
                if "weight" in name:
                    print(name, param.size(), param.shape, sum(p.numel() for p in param))
                    ff += sum(p.numel() for p in param)

        print("pruning.py: stats: number of elements for each kind of layer:")
        print("pruning.py: stats: Conv2d: ", conv2d)
        print("pruning.py: stats: Conv1d: ", conv1d)
        print("pruning.py: stats: ff: ", ff)
        print("pruning.py: stats:: mha: ", mha)
        #print(name, type(param), param.size(), param.shape, param[:2])
        print("pruning.py: stats: overall model parameters", sum(p.numel() for p in model.parameters()))
        #torch.set_printoptions(profile="default")
        print("*********************")

        Cv0 = []
        Cv2 = []
        for name, param in model.named_parameters():
            if "encoder.embed.conv.0.weight" in name:
                print(name, param.size())
                #print("******", param)
                #Mw1.append[torch.mean(torch.abs((torch.flatten(param))))]
                w0 = torch.mean(torch.abs((torch.flatten(param))))
                Cv0.append(w0.detach().numpy())
            elif "encoder.embed.conv.2.weight" in name:
                print(name, param.size())
                w2 = torch.mean(torch.abs((torch.flatten(param))))
                Cv2.append(w2.detach().numpy())
        print("pruning.py: stats: Cv0", Cv0)
        print("pruning.py: stats: Cv2", Cv2)

        Cv1 = []
        Cv = []
        Cv3 = []
        for name, param in model.named_parameters():
            if "conv_module.pointwise_conv1.weight" in name:
                print(name, param.size())
                w1 = torch.mean(torch.abs((torch.flatten(param))))
                Cv1.append(w1.detach().numpy())
            elif "conv_module.depthwise_conv.weight" in name:
                print(name, param.size())
                w2 = torch.mean(torch.abs((torch.flatten(param))))
                Cv.append(w2.detach().numpy())
            elif "conv_module.pointwise_conv2.weight" in name:
                print(name, param.size())
                w3 = torch.mean(torch.abs((torch.flatten(param))))
                Cv3.append(w3.detach().numpy())
        print("pruning.py: stats: Cv1", Cv1)
        print("pruning.py: stats: Cv", Cv)
        print("pruning.py: stats: Cv3", Cv3)

        FF1 = []
        FF2 = []
        FF3 = []
        FF4 = []
        for name, param in model.named_parameters():
            if "feed_forward.w_1.weight" in name:
                print(name, param.size())
                #print("******", param)
                #Mw1.append[torch.mean(torch.abs((torch.flatten(param))))]
                w1 = torch.mean(torch.abs((torch.flatten(param))))
                FF1.append(w1.detach().numpy())
            elif "feed_forward.w_2.weight" in name:
                print(name, param.size())
                w2 = torch.mean(torch.abs((torch.flatten(param))))
                FF2.append(w2.detach().numpy())
            elif "feed_forward_macaron.w_1.weight" in name:
                print(name, param.size())
                w3 = torch.mean(torch.abs((torch.flatten(param))))
                FF3.append(w3.detach().numpy())
            elif "feed_forward_macaron.w_2.weight" in name:
                print(name, param.size())
                w4 = torch.mean(torch.abs((torch.flatten(param))))
                FF4.append(w4.detach().numpy())
        print("pruning.py: stats: FF1", FF1)
        print("pruning.py: stats: FF2", FF2)                       
        print("pruning.py: stats: FF3", FF3)                        
        print("pruning.py: stats: FF4", FF4)                       
       
        Swq = []
        Swk = []
        Swv = []
        Swo = []
        Swp = []
        for name, param in model.named_parameters():
            if "self_attn.linear_q.weight" in name:
                print(name, param.size())
                w1 = torch.mean(torch.abs((torch.flatten(param))))
                Swq.append(w1.detach().numpy())
            elif "self_attn.linear_k.weight" in name:
                print(name, param.size())
                w2 = torch.mean(torch.abs((torch.flatten(param))))
                Swk.append(w2.detach().numpy())
            elif "self_attn.linear_v.weight" in name:
                print(name, param.size())
                w3 = torch.mean(torch.abs((torch.flatten(param))))
                Swv.append(w3.detach().numpy())
            elif "self_attn.linear_out.weight" in name:
                print(name, param.size())
                w4 = torch.mean(torch.abs((torch.flatten(param))))
                Swo.append(w4.detach().numpy())
            elif "self_attn.linear_pos.weight" in name:
                print(name, param.size())
                w5 = torch.mean(torch.abs((torch.flatten(param))))
                Swp.append(w5.detach().numpy())
        print("pruning.py: stats: Swk", Swk)                        
        print("pruning.py: stats: Swq", Swq)                       
        print("pruning.py: stats: Swv", Swv)                        
        print("pruning.py: stats: Swo", Swo)                       
        print("pruning.py: stats: Swp", Swp)                       

        """
        occ1_w = np.zeros((enc_ff1.size(dim=0), bins))
        enc_ff = enc_ff1
        for i in range(0, enc_ff.size(dim=0)):
            for j in range(0, enc_ff[i].size(dim=0)):
                for k in range(0, bins-1):
                    if enc_ff[i][j] >= marge1[i][k] and enc_ff[i][j] < marge1[i][k+1]:
                        occ1_w[i][k] +=1
        """        
  
        
    if args.prune_asr_model:
        ok = 0

        logging.info("args.prune_asr_model: {args.prune_asr_model}")
        if args.verbose:

            print("args.prune_asr_model:",args.prune_asr_model)
            print("args.prune_asr_local:",args.prune_asr_model_local)
            print("args.prune_asr_adapt:",args.prune_asr_model_adapt)
            print("args.prune_asr_tile_bc:",args.prune_asr_model_tile_bc)
        
        if args.prune_asr_model_local:
            print("prune_asr_model_local: FF rate=",args.am," Attention rate=",args.amAtt)
            # let's try to see if we can compute a sparsity:
            nzeros=ntotal=0
            for name, module in model.named_modules():
                if ("encoder.encoders." in str(name) or "decoder.decoders." in str(name)):
                    if isinstance(module,(torch.nn.Linear)):
                        nzeros += float(torch.sum(module.weight == 0))
                        ntotal += float(module.weight.nelement())
            print("prune_asr_model_local: sparsity for all linear modules before local pruning:", 100 * nzeros/ntotal)
            
            am = args.am
            amAtt = args.amAtt
            sp_end = n_elt = sp_ini = 0

            # JLR modif pour des taux de prining différents entre FF et attention 
            for name, module in model.named_modules():
                if ("encoder.encoders." in str(name) or "decoder.decoders." in str(name)):
                    if("feed_forward" in str(name)):
                        if isinstance(module,(torch.nn.Linear)):
                            if args.verbose: print("pruned module: name ",name," module ",module)
                            sp_ini +=  float(torch.sum(module.weight == 0))
                            prune.l1_unstructured(module, name='weight', amount=am)
                            prune.remove(module, 'weight')
                            sp_end += float(torch.sum(module.weight == 0))
                            n_elt += float(module.weight.nelement())
                    #if("self_attn" in str(name)):
                    if("attn" in str(name)):
                        if isinstance(module,(torch.nn.Linear)):
                            if args.verbose: print("pruned module: name ",name," module ",module)
                            sp_ini +=  float(torch.sum(module.weight == 0))
                            prune.l1_unstructured(module, name='weight', amount=amAtt)
                            prune.remove(module, 'weight')
                            sp_end += float(torch.sum(module.weight == 0))
                            n_elt += float(module.weight.nelement())

            print("prune_asr_model_local: initial sparsity", 100 * sp_ini/n_elt)
            print("prune_asr_model_local: final sparsity", 100 * sp_end/n_elt, "(",sp_end,"/",n_elt,")")



            
            
        if args.prune_asr_model_adapt:
            print("prune_asr_model_adapt")
            enc_ff_rate = args.enc
            dec_ff_rate = args.dec
            all_att_rate = args.fixe

 
            sp_ini = sp_end = n_elt = 0

            print("prune_asr_model_adapt (varible scale pruning): att rate:",all_att_rate,"initial enc rate",enc_ff_rate,"initial dec rate",dec_ff_rate)

            frate=0.9
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if args.verbose: print("prune_asr_model_adapt: name:",name,"module:",module)
                    if "encoder" in name:
                        if ("feed_forward." in name):
                            sp_ini +=  float(torch.sum(module.weight == 0)) 
                            prune.l1_unstructured(module, name='weight', amount=enc_ff_rate)
                            prune.remove(module, 'weight')
                            sp_end +=  float(torch.sum(module.weight == 0)) 
                            n_elt += float(module.weight.nelement())
                            spars = float(torch.sum(module.weight == 0))/float(module.weight.nelement())
                            num=name.split(".")
                            numblock=num[2]
                            if args.verbose: print("prune_asr_model_adapt: pruned module: ",name," module",module, "enc_ff_rate",enc_ff_rate, "spars",spars)
                            # decrease rate only after w2 layer (because we have w1 and w2... )
                            if("w_2" in name):
                                print("prune_asr_model_adapt: encoder block:",numblock," decrease encoder ff pruning rate:",enc_ff_rate,"-0.01")
                                enc_ff_rate -= 0.01

                        if("attn" in name):
                            sp_ini +=  float(torch.sum(module.weight == 0))
                            prune.l1_unstructured(module, name='weight', amount=all_att_rate)
                            prune.remove(module, 'weight')
                            sp_end +=  float(torch.sum(module.weight == 0))
                            n_elt += float(module.weight.nelement())
                            spars = float(torch.sum(module.weight == 0))/float(module.weight.nelement())
                            if args.verbose: print("prune_asr_model_adapt: prune module:",name," module",module, "all_att_rate",all_att_rate, "spars",spars)
                        
                    if "decoder" in name:
                        if args.verbose: print("prune_asr_model_adapt: decoder module:",module)

                        if ("feed_forward." in name):
                            sp_ini +=  float(torch.sum(module.weight == 0))
                            prune.l1_unstructured(module, name='weight', amount=dec_ff_rate)
                            prune.remove(module, 'weight')
                            sp_end +=  float(torch.sum(module.weight == 0))
                            n_elt += float(module.weight.nelement())
                            spars = float(torch.sum(module.weight == 0))/float(module.weight.nelement())
                            num=name.split(".")
                            numblock=num[2]
                            if args.verbose: print("prune_asr_model_adapt: pruned module:",name," module",module, "dec_ff_rate",dec_ff_rate, "spars",spars)
                            # decrease rate only after w2 layer (because we have w1 and w2... )
                            if("w_2" in name):
                                print("prune_asr_model_adapt: decoder block:",numblock,"decrease decoder ff pruning rate:",dec_ff_rate,"-0.01")
                                dec_ff_rate -= 0.01
                        
                        if("attn" in name):
                            sp_ini +=  float(torch.sum(module.weight == 0))
                            prune.l1_unstructured(module, name='weight', amount=all_att_rate)
                            prune.remove(module, 'weight')
                            sp_end +=  float(torch.sum(module.weight == 0))
                            n_elt += float(module.weight.nelement())
                            spars = float(torch.sum(module.weight == 0))/float(module.weight.nelement())
                            if args.verbose: print("prune_asr_model_adapt: pruned module:",name," module",module, "all_att_rate",all_att_rate, "spars",spars)

            print("prune_asr_model_adapt: initial sparsity", 100 * sp_ini/n_elt)
            print("prune_asr_model_adapt: final sparsity", 100 * sp_end/n_elt)

##################################################################
        if args.prune_asr_model_tile_bc:
            print("prune_asr_model_tile_bc")
            # let's try to see if we can compute a sparsity:
            nzeros=ntotal=0
            for name, module in model.named_modules():
                if ("encoder.encoders." in str(name) or "decoder.decoders." in str(name)):
                    if isinstance(module,(torch.nn.Linear)):
                        nzeros += float(torch.sum(module.weight == 0))
                        ntotal += float(module.weight.nelement())
            print("prune_asr_model_tile_bc: initial sparsity:", 100 * nzeros/ntotal)

            n = args.tile
            thres = args.thres
            am = args.am
            sp_end = n_elt = 0
            ntiles=0  
            if args.tileFF is True:
                print("prune_asr_model_tile_bc: tiles computed only on FF layers")
                part="feed_forward"
            else:
                part="coder" # should work with all ?

            model_tiles=0
            for name, param in model.named_parameters():
                if (part in name and "weight" in name) and ("conv" not in name) and ("norm" not in name):
                    if args.verbose: print(name,"matrix size : ",param.shape[0], " * ", param.shape[1],)
                    layer_ntiles =  0
                    moy_param=torch.mean(torch.abs(param))
                    for i in range(0, param.shape[0] , n):
                        for j in range(0, param.shape[1] , n):
                            moy=torch.mean(torch.abs(param[i:i+n,j:j+n]))
                            #model_tiles +=1
                            if moy < thres * moy_param:
                                with torch.no_grad():
                                    param[i:i+n,j:j+n] = torch.zeros(n,n)
                                ntiles += 1
                                layer_ntiles += 1
        
                    if args.verbose: print("prune_asr_model_tile_bc: module name: ", name,"matrix size : ",param.shape[0], " * ", param.shape[1],"current number of tiles", model_tiles)
                    total = (param.shape[0]*param.shape[1])/(n*n)
                    model_tiles +=total
                    print("prune_asr_model_tile_bc:",name,"# tiles:",total,"pruned tiles", layer_ntiles,"tiles pruning % ",100*layer_ntiles/total)

            print("prune_asr_model_tile_bc: model: number of tiles", model_tiles,"number of pruned ntiles=    ",ntiles,"pruned tiles  %:", ntiles/model_tiles)

            # compute sparsity for each layer
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear)):
                    sp_end += float(torch.sum(module.weight == 0))
                    n_elt += float(module.weight.nelement())
                    sp = float(torch.sum(module.weight == 0))
                    n = float(module.weight.nelement())
                    spars = sp/n
                    if args.verbose: print("prune_asr_model_tile_bc: module ", name ," ", module, "spars : ", spars)

            print("prune_asr_model_tile_bc: local mean tiles sparsity", 100 * sp_end/n_elt,"(",sp_end,"/",n_elt,")")

            # let's try to see if we can compute a sparsity:
            nzeros=ntotal=0
            for name, module in model.named_modules():
                if ("encoder.encoders." in str(name) or "decoder.decoders." in str(name)):
                    if isinstance(module,(torch.nn.Linear)):
                        nzeros += float(torch.sum(module.weight == 0))
                        ntotal += float(module.weight.nelement())
            print("prune_asr_model_tile_bc: final sparsity:", 100 * nzeros/ntotal)
    ########## end prune_asr_model_tile_bc #######################


##################################################################
        if args.prune_asr_model_tile_percent:
            print("prune_asr_model_tile_percent")
            # let's try to see if we can compute a sparsity:
            nzeros=ntotal=0
            for name, module in model.named_modules():
                if ("encoder.encoders." in str(name) or "decoder.decoders." in str(name)):
                    if isinstance(module,(torch.nn.Linear)):
                        nzeros += float(torch.sum(module.weight == 0))
                        ntotal += float(module.weight.nelement())
            print("prune_asr_model_tile_percent: initial sparsity:", 100 * nzeros/ntotal)

            n = args.tile
            thres = args.thres
            am = args.am
            sp_end = n_elt = 0
            ntiles=0  
            if args.tileFF is True:
                print("prune_asr_model_tile_percent: tiles computed only on FF layers")
                part="feed_forward"
            else:
                part="coder" # should work with all ?

            model_tiles=0
            model_zerotiles=0
            for name, param in model.named_parameters():
                if (part in name and "weight" in name) and ("conv" not in name) and ("norm" not in name):
                    if args.verbose: print(name,"matrix size : ",param.shape[0], " * ", param.shape[1], "=",param.shape[0]*param.shape[1],"->",param.shape[0]*param.shape[1]/(n*n),"tiles")
                    blockmoy=np.zeros(int(param.shape[0]*param.shape[1]/(n*n)))
                    layer_ntiles =  0
                    layer_zerotiles = 0
                    
                    moy_param=torch.mean(torch.abs(param))
                    #print("number of zero?",np.count_nonzero(param==0))
                    start_time = time.time()
                    for i in range(0, param.shape[0], n):
                        for j in range(0, param.shape[1], n):
                            moy=torch.mean(torch.abs(param[i:i+n,j:j+n]))
                            #print(moy.item())
                            # store moy for each tile for each layer
                            splitname=name.split(".")
                            blocknumber=splitname[2]
                            blockmoy[layer_ntiles]=moy.item()
                            ntiles += 1
                            layer_ntiles += 1

                    end_time = time.time()
                    execution_time = end_time - start_time
                    #print("Execution time:", execution_time, "seconds")
                
                    # Step 1: Sort the vector
                    v_sorted = np.sort(blockmoy)
                    #print("zeros.",np.count_nonzero(v_sorted == 0))
                    percent=thres
                    # Step 2: Calculate the index for the Xth percentile
                    percentile_index = int(percent * len(v_sorted))
                    # Step 3: Retrieve the threshold value
                    threshold = v_sorted[percentile_index]
                    # check 
                    print("nbelement under threshold",np.count_nonzero(blockmoy <= threshold),"percentage=",100*np.count_nonzero(blockmoy <= threshold)/len(blockmoy))
                    if args.verbose: print("Threshold for the",percent*100,"% lowest elements:", threshold, "index",percentile_index,"/",int(param.shape[0]*param.shape[1]/n*n),"layer moy",moy_param)
                    realtotal=0
                    for i in range(0, param.shape[0], n):
                        for j in range(0, param.shape[1], n):
                            realtotal+=1
                            moy=torch.mean(torch.abs(param[i:i+n,j:j+n]))
                            if moy.item() <= threshold: 
                                with torch.no_grad():
                                    param[i:i+n,j:j+n] = torch.zeros(n,n)
                                layer_zerotiles+=1
                    print("real number of tiles",realtotal)
                    #print("tiles number :",layer_ntiles,"zero tiles",layer_zerotiles,"percent",layer_zerotiles/layer_ntiles,"consigne",percent,"threshold",threshold)

                    
                    if args.verbose: print("prune_asr_model_tile_percent: module name: ", name,"matrix size : ",param.shape[0], " * ", param.shape[1],"current number of tiles", model_tiles)
                    total = (param.shape[0]*param.shape[1])/(n*n)
                    model_tiles +=total
                    model_zerotiles+=layer_zerotiles
                    print("prune_asr_model_tile_percent:",name,"# tiles:",total,"pruned tiles", layer_zerotiles,"tiles pruning % ",100*layer_zerotiles/total)
            
            print("prune_asr_model_tile_percent: model: number of tiles", model_tiles,"number of pruned ntiles=    ",model_zerotiles,"pruned tiles  %:", model_zerotiles/model_tiles)

            # compute sparsity for each layer
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear)):
                    sp_end += float(torch.sum(module.weight == 0))
                    n_elt += float(module.weight.nelement())
                    sp = float(torch.sum(module.weight == 0))
                    n = float(module.weight.nelement())
                    spars = sp/n
                    if args.verbose: print("prune_asr_model_tile_percent: module ", name ," ", module, "spars : ", spars)

            print("prune_asr_model_tile_percent: local mean tiles sparsity", 100 * sp_end/n_elt,"(",sp_end,"/",n_elt,")")

            # let's try to see if we can compute a sparsity:
            nzeros=ntotal=0
            for name, module in model.named_modules():
                if ("encoder.encoders." in str(name) or "decoder.decoders." in str(name)):
                    if isinstance(module,(torch.nn.Linear)):
                        nzeros += float(torch.sum(module.weight == 0))
                        ntotal += float(module.weight.nelement())
            print("prune_asr_model_tile_percent: final sparsity:", 100 * nzeros/ntotal)
    ########## end prune_asr_model_tile_percent #######################

    if args.prune_asr_model_tile_round:
            print("prune_asr_model_tile_round")
            # let's try to see if we can compute a sparsity:
            nzeros=ntotal=0
            for name, module in model.named_modules():
                if ("encoder.encoders." in str(name) or "decoder.decoders." in str(name)):
                    if isinstance(module,(torch.nn.Linear)):
                        nzeros += float(torch.sum(module.weight == 0))
                        ntotal += float(module.weight.nelement())
            print("prune_asr_model_tile_round: initial sparsity:", 100 * nzeros/ntotal)

            n = args.tile
            thres = args.thres
            am = args.am
            sp_end = n_elt = 0
            ntiles=0  
            if args.tileFF is True:
                print("prune_asr_model_tile_round: tiles computed only on FF layers")
                part="feed_forward"
            else:
                part="coder" # should work with all ?

            model_tiles=0
            model_zerotiles=0
            for name, param in model.named_parameters():
                if (part in name and "weight" in name) and ("conv" not in name) and ("norm" not in name):
                    if args.verbose: print(name,"matrix size : ",param.shape[0], " * ", param.shape[1], "=",param.shape[0]*param.shape[1],"->",param.shape[0]*param.shape[1]/(n*n),"tiles")
                    blockmoy=np.zeros(int(param.shape[0]*param.shape[1]/(n*n)))
                    layer_ntiles =  0
                    layer_zerotiles = 0
                    
                    moy_param=torch.mean(torch.abs(param))
                    #print("number of zero?",np.count_nonzero(param==0))
                    start_time = time.time()
                    # Your optimized loop here
                    for i in range(0, param.shape[0] , n):
                        for j in range(0, param.shape[1] , n):
                            moy=torch.mean(torch.abs(param[i:i+n,j:j+n]))
                            #print(moy.item())
                            # store moy for each tile for each layer
                            splitname=name.split(".")
                            blocknumber=splitname[2]
                            blockmoy[layer_ntiles]=moy.item()
                            ntiles += 1
                            layer_ntiles += 1

                    end_time = time.time()
                    execution_time = end_time - start_time
                    #print("Execution time:", execution_time, "seconds")
                
                    # Step 1: Sort the vector
                    v_sorted = np.sort(blockmoy)
                    #print("zeros.",np.count_nonzero(v_sorted == 0))
                    percent=thres
                    # Step 2: Calculate the index for the Xth roundile
                    percentile_index = int(percent * len(v_sorted))
                    # Step 3: Retrieve the threshold value
                    threshold = v_sorted[percentile_index]
                    #print("Threshold for the",round*100,"% lowest elements:", threshold, "index",roundile_index,"/",int(param.shape[0]*param.shape[1]/n*n),"layer moy",moy_param)
                    for i in range(0, param.shape[0] , n):
                        for j in range(0, param.shape[1] , n):
                            moy=torch.mean(torch.abs(param[i:i+n,j:j+n]))
                            if moy.item() < threshold: 
                                with torch.no_grad():
                                    param[i:i+n,j:j+n] = torch.zeros(n,n)
                                layer_zerotiles+=1
                            else:
                                with torch.no_grad():
                                    # it is ok to do that ? 
                                    #param[i:i+n,j:j+n] = torch.round(param[i:i+n,j:j+n]*2)/2
                                    # is it better to do this ? 
                                    param[i:i+n,j:j+n] = torch.round(moy*10)/10
                                    if moy.item()==0:
                                        layer_zerotiles+=1
                    #print("tiles number :",layer_ntiles,"zero tiles",layer_zerotiles,"round",layer_zerotiles/layer_ntiles,"consigne",round,"threshold",threshold)
                    
                    # c'est pour compter ! 
                    k=0
                    val=np.zeros(int(param.shape[0]*param.shape[1]/(n*n)))
                    for i in range(0, param.shape[0] - n, n):
                        for j in range(0, param.shape[1] - n, n):
                            value=param[i,j]
                            val[k]=value.item()
                            k+=1
                    # comptage des valeurs????
                    count_dict = Counter(val)
                    # Afficher les résultats
                    for value, count in count_dict.items():
                        print("Valeur:", value, "- Nombre d'éléments:", count)
                    if args.verbose: print("prune_asr_model_tile_round: module name: ", name,"matrix size : ",param.shape[0], " * ", param.shape[1],"current number of tiles", model_tiles)
                    total = (param.shape[0]*param.shape[1])/(n*n)
                    model_tiles +=total
                    model_zerotiles+=layer_zerotiles
                    print("prune_asr_model_tile_round:",name,"# tiles:",total,"pruned tiles", layer_zerotiles,"tiles pruning % ",100*layer_zerotiles/total)
            
            print("prune_asr_model_tile_round: model: number of tiles", model_tiles,"number of pruned ntiles=    ",model_zerotiles,"pruned tiles  %:", model_zerotiles/model_tiles)

            # compute sparsity for each layer
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear)):
                    sp_end += float(torch.sum(module.weight == 0))
                    n_elt += float(module.weight.nelement())
                    sp = float(torch.sum(module.weight == 0))
                    n = float(module.weight.nelement())
                    spars = sp/n
                    if args.verbose: print("prune_asr_model_tile_round: module ", name ," ", module, "spars : ", spars)

            print("prune_asr_model_tile_round: local mean tiles sparsity", 100 * sp_end/n_elt,"(",sp_end,"/",n_elt,")")

            # let's try to see if we can compute a sparsity:
            nzeros=ntotal=0
            for name, module in model.named_modules():
                if ("encoder.encoders." in str(name) or "decoder.decoders." in str(name)):
                    if isinstance(module,(torch.nn.Linear)):
                        nzeros += float(torch.sum(module.weight == 0))
                        ntotal += float(module.weight.nelement())
            print("prune_asr_model_tile_round: final sparsity:", 100 * nzeros/ntotal)
    ########## end prune_asr_model_tile_round #######################

    
            
    # let's try to see if we can compute a sparsity:
    nzeros=ntotal=0
    for name, module in model.named_modules():
        if ("encoder.encoders." in str(name) or "decoder.decoders." in str(name)):
            if isinstance(module,(torch.nn.Linear)):
                nzeros += float(torch.sum(module.weight == 0))
                ntotal += float(module.weight.nelement())
    print("prune_transformer: final sparsity (encoders+decoders layers):", 100 * nzeros/ntotal, "(",nzeros,"/",ntotal,")")

    # let's try to see if we can compute a sparsity:
    nzeros=ntotal=0
    for name, module in model.named_modules():
        if isinstance(module,(torch.nn.Linear)):
            nzeros += float(torch.sum(module.weight == 0))
            ntotal += float(module.weight.nelement())
    print("prune_transformerc: final sparsity (all linear layers):", 100 * nzeros/ntotal)

    if args.save_to:
        print("modelprune: model saved in ",args.save_to)
        torch.save(model.state_dict(),args.save_to)

        # quantize model for saving and size measurement...        
        print("modelprune: quantize")
        q_config = {torch.nn.Linear} #Conv2d, torch.nn.Conv1d}
        dtype = getattr(torch, "qint8")
        modelq = torch.quantization.quantize_dynamic(model, q_config, dtype=dtype)
        torch.save(modelq.state_dict(),args.save_to+".quant")
        size=os.path.getsize(args.save_to+".quant")
        print("pruned quantized model size: ", size/1e3)

        




        
#############################
# from espnet/asr/pytorch_backend/asr_init.py
def load_trained_model(model_path, training=True):
    """Load the trained model for recognition.

    Args:
        model_path (str): Path to model.***.best
        training (bool): Training mode specification for transducer model.

    Returns:
        model (torch.nn.Module): Trained model.
        train_args (Namespace): Trained model arguments.

    """
    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), "model.json")
    )

    logging.info(f"Reading model parameters from {model_path}")
    
    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"

    # CTC Loss is not needed, default to builtin to prevent import errors
    if hasattr(train_args, "ctc_type"):
        train_args.ctc_type = "builtin"

    model_class = dynamic_import(model_module)

    if "transducer" in model_module:
        model = model_class(idim, odim, train_args, training=training)
        custom_torch_load(model_path, model, training=training)
    else:
        model = model_class(idim, odim, train_args)
        torch_load(model_path, model)
        
    return model, train_args


###############@

# from espnet/asr/asr_utils.py
#
# * -------------------- general -------------------- *
def get_model_conf(model_path, conf_path=None):
    """Get model config information by reading a model config file (model.json).

    Args:
        model_path (str): Model path.
        conf_path (str): Optional model config path.

    Returns:
        list[int, int, dict[str, Any]]: Config information loaded from json file.

    """

    if conf_path is None:
        model_conf = os.path.dirname(model_path) + "/model.json"
    else:
        model_conf = conf_path

    with open(model_conf, "rb") as f:
        logging.info("reading a config file from " + model_conf)
        confs = json.load(f)
    if isinstance(confs, dict):
        # for lm
        args = confs
        return argparse.Namespace(**args)
    else:
        # for asr, tts, mt
        idim, odim, args = confs
        return idim, odim, argparse.Namespace(**args)


def torch_load(path, model):
    """Load torch model states.

    Args:
        path (str): Model path or snapshot file path to be loaded.
        model (torch.nn.Module): Torch model.

    """
    if "snapshot" in os.path.basename(path):
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)[
            "model"
        ]
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)

    if hasattr(model, "module"):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict


    
#from espnet.utils.dynamic_import import dynamic_import
def dynamic_import(import_path, alias=dict()):
    """dynamic import module and class

    :param str import_path: syntax 'module_name:class_name'
        e.g., 'espnet.transform.add_deltas:AddDeltas'
    :param dict alias: shortcut for registered class
    :return: imported class
    """
    if import_path not in alias and ":" not in import_path:
        raise ValueError(
            "import_path should be one of {} or "
            'include ":", e.g. "espnet.transform.add_deltas:AddDeltas" : '
            "{}".format(set(alias), import_path)
        )
    if ":" not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(":")
    m = importlib.import_module(module_name)
    return getattr(m, objname)
