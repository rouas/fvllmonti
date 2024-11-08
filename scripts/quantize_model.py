#!/usr/bin/env python3

import torch
import os
import copy

# new addidtion 
import numpy as np
import pandas as pd
import json
import itertools

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

#         print("Dynamic quantization ******************")
#         q_config = {torch.nn.Linear} #Conv2d, torch.nn.Conv1d}
#         dtype = getattr(torch, "qint8")
#         model = torch.quantization.quantize_dynamic(m, q_config, dtype=dtype)
# #        print("lbz: pruned_quantized_model           ",model)
#         torch.save(model.state_dict(),"test.quant.statedict")
#         torch.save(model, "test.quant") # donc il faut ouvrir le fichier crée qui est aussi très grand
#         size=os.path.getsize("test.quant")

#         output_dir="temp-dynamic"
#         for name, module in model.named_modules():
#                 # Check if the parameter is a weight tensor
#                 print("name=",name) #," module=",module)
#                 # does not work for convolution layers.... 
#                 if isinstance(module, torch.nn.Linear): #QuantizedModule): #torch.nn.Linear): #DynamicQuantizedLinear): #torch.nn.Linear):
#                         if "ctc" not in name and "embed" not in name:
#                                 file_name = f"{output_dir}_txt/weights_{name}.txt"
#                                 #print("Saving module weights to:", file_name)
#                                 #np.savetxt(file_name, module.weight.numpy())
#                                 np.savetxt(file_name, module.weight.detach().numpy())

#         print("Process completed.")
        
#         print("dynamic  args.quantize_asr_model lbz: pruned quantized model size: ", size/1e3)
#         print("dynamic  recog: args.quantize_asr_model: model saved in test.quant")	
#         print("*****************************************************")
        

        print("Post-training static quantization *******************************")



        print("  from https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html")
        from torch.ao.quantization import get_default_qconfig
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        from torch.ao.quantization import QConfigMapping

        from espnet.utils.io_utils import LoadInputsAndTargets
        load_inputs_and_targets = LoadInputsAndTargets(
                mode="asr",
                load_output=True,
                sort_in_input_length=False,
                preprocess_conf=("/vol/experiments/rouas/Fvllmonti/srvrouas2/librispeech/asrjl/conf/specaug.yaml"),
#                        train_args.preprocess_conf
#                        if args.preprocess_conf is None
 #                       else args.preprocess_conf
  #              ),
                preprocess_args={"train": True},
        )

        float_model=copy.deepcopy(m)
        float_model.eval()
        # The old 'fbgemm' is still available but 'x86' is the recommended default.
        qconfig = get_default_qconfig("x86")
        qconfig_mapping = QConfigMapping().set_global(qconfig)

        batch = "../librispeech/asrjl/dump/test_clean/deltafalse/split16utt/data_unigram5000.1.json"
        jsonfile="../librispeech/asrjl/dump/test_clean/deltafalse/split16utt/data_unigram5000.1.json"

        #"../libri_trans/asrjl-new-fromscratch/dump/train_dev.en/deltafalse/data_bpe1000.lc.rm.json" #[(name, js[name])]
        # batch = [('utts',
        #         dict(input=[dict(feat='some.ark:123',filetype='mat',name='input1', shape=[100, 80])],
        #         output=[dict(tokenid='1 2 3 4',name='target1',shape=[4, 31])]]
        # # read json data
        with open(jsonfile, "rb") as f:
            js = json.load(f)["utts"]
        with torch.no_grad():
                for idx, name in enumerate(js.keys(), 1):
                        print("(%d/%d) decoding " + name, idx, len(js.keys()))
                        batch = [(name, js[name])]
                        feat = load_inputs_and_targets(batch)[0][0]
                        print(feat.shape)
        #feat = load_inputs_and_targets(js)
        #[0][0]
        #print(feat.size)

        def grouper(n, iterable, fillvalue=None):
                kargs = [iter(iterable)] * n
                return itertools.zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        feat_lens = [js[key]["input"][0]["shape"][0] for key in keys]
        sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
        keys = [keys[i] for i in sorted_index]

        batchsize=10
        with torch.no_grad():
                for names in grouper(batchsize, keys, None):
                        names = [name for name in names if name]
                        batch = [(name, js[name]) for name in names]
                        feats = (
                               load_inputs_and_targets(batch)
                        )

                        print(len(feats))
                        fea = feats[0]
                        print(fea)
                        def calibrate(model, data_loader):
                                model.eval()
                                with torch.no_grad():
                                        for image, target in data_loader:
                                                model(image)

        example_inputs = feat #(next(iter(data_loader))[0]) # get an example input
        float_model(example_inputs)
        prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)  # fuse modules and insert observers
        calibrate(prepared_model, data_loader_test)  # run calibration on sample data
        quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model

        ##############################@x


        
        # from https://pytorch.org/blog/quantization-in-practice/#post-training-static-quantization-ptq
        ## EAGER MODE
        mq = copy.deepcopy(m)
        mq.eval()

        """Fuse
        - Inplace fusion replaces the first module in the sequence with the fused module, and the rest with identity modules
        """
        torch.quantization.fuse_modules(mq, ['0','1'], inplace=True) # fuse first Conv-ReLU pair
        torch.quantization.fuse_modules(mq, ['2','3'], inplace=True) # fuse second Conv-ReLU pair
        
        """Insert stubs"""
        mq = nn.Sequential(torch.quantization.QuantStub(), 
                          *mq, 
                          torch.quantization.DeQuantStub())

        

        backend = "qnnpack" #"x86" #fbgemm" #x86"
        #m.qconfig = torch.quantization.get_default_qconfig(backend)
        model_static_quantized=m
        model_static_quantized.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        model_static_quantized = torch.quantization.prepare(m, inplace=False)
        model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
        torch.save(model_static_quantized, "test_static.quant") # donc il faut ouvrir le fichier crée qui est aussi très grand
        torch.save(model_static_quantized.state_dict(),"test_static.quant.statedict")
        
#        torch.quantization.prepare(m, inplace=True)
#        mq=torch.quantization.convert(m, inplace=False)
#        torch.save(m, "test_static.quantinplace") # donc il faut ouvrir le fichier crée qui est aussi très grand
#        torch.save(m.state_dict(),"test_static.quantinplace.statedict")

        output_dir="temp"
        for name, module in model_static_quantized.named_modules():
                # Check if the parameter is a weight tensor
                print(name)
                # does not work for convolution layers.... 
                if isinstance(module, torch.nn.Linear):
                        if "ctc" not in name and "embed" not in name:
                                file_name = f"{output_dir}_txt/weights_{name}.txt"
                                #print("Saving module weights to:", file_name)
                                #np.savetxt(file_name, module.weight.numpy())
                                np.savetxt(file_name, module.weight.detach().numpy())

        print("Process completed.")
        

        
if __name__ == "__main__":
        main(sys.argv[1:])

                
                
