#!/usr/bin/env python3

import torch
import numpy as np
import os
import sys

# for loading... 
import logging
import json
import argparse
import importlib


def save_module_weights_txt(model_path, output_dir):
        # Check if the provided model path exists
        if not os.path.exists(model_path):
                print("The specified model path doesn't exist.")
                return

        # Load the model
        #m = torch.load(model_path)
        # need that for loading state dict saved models.... 
        m, train_args = load_trained_model(model_path, training=False)

        print("model loaded")
        torch.set_printoptions(profile="full")  # To display all values

        # np.savetxt("test.txt",m)
        # for name, module in m.named_modules():
        #         if isinstance(module, torch.nn.Linear):
        #                 if "ctc" not in name and "embed" not in name:
        #                         print(name)
        #                         file_name = os.path.join(output_dir, "SavedModules_" + name + ".txt")
        #                         print("Saving module weights to:", file_name)
        #                         np.savetxt(file_name, module.weight.numpy())
        for name, module in m.named_modules():
        # Check if the parameter is a weight tensor
                #print(name)
                # does not work for convolution layers.... 
                if isinstance(module, torch.nn.Linear):
                        if "ctc" not in name and "embed" not in name:
                                file_name = f"{output_dir}_txt/weights_{name}.txt"
                                #print("Saving module weights to:", file_name)
                                #np.savetxt(file_name, module.weight.numpy())
                                np.savetxt(file_name, module.weight.detach().numpy())

        print("Process completed.")


import pickle
# pickle version
def save_module_weights_pickle(model_path, output_path):
        torch.set_printoptions(profile="full")  # To display all values
#        m = torch.load(model_path)
        m, train_args = load_trained_model(model_path, training=False)

        # Extract model state dictionary
        state_dict = m.state_dict()
        # Save the state dictionary using pickle
        with open(output_path+".pkl", 'wb') as f:
                pickle.dump(state_dict, f)



# import torch
# import os

# # new addidtion 
# import numpy as np
# import pandas as pd



# m = torch.load("saved_models/pruned_tiles_quantized/The_model_complet")
# torch.set_printoptions(profile="full") # Pour afficher toutes les valeurs
# for name, module in m.named_modules():
#         if isinstance(module, torch.nn.Linear):
#                 if not "ctc" in name and not "embed" in name:

#                         print(name)
#                         nomfich="/tmp/SavedModules_" + name
#                         print(nomfich)
#                         torch.save(module,nomfich)
#                         s=torch.load(nomfich)
#                         v=s.weight
#                         # instead 
#                         nomfich2="SavedModules_" + name + ".txt"

#                         np.savetxt(nomfich2,v.numpy(force=True))

#                         #np.savetxt(nomfich,intversion.numpy())
#                         #np.savetxt(nomfich,v.numpy())



# this works but save in int8 format
#s=torch.load("saved_models/pruned_tiles_quantized/SavedModules_encoder.encoders.4.self_attn.linear_k")
#v=s.weight()
#intversion=torch.int_repr(v)
#np.savetxt("t",intversion.numpy())
        




        
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





if __name__ == "__main__":
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_path> <output_dir>")
    else:
        model_path = sys.argv[1]
        output_dir = sys.argv[2]

        save_module_weights_txt(model_path,output_dir)
        # alternate pickle
        save_module_weights_pickle(model_path,output_dir)
        ####

        #save_module_weights(model_path, output_dir)
