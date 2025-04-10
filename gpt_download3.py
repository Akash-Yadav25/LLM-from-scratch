import os
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    #load setting and params
    print(os.path.join(models_dir, "hparams.json"))
    tf_ckpt_path = tf.train.latest_checkpoint(models_dir)
    settings = json.load(open(os.path.join(models_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path,settings)
    return settings, params

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    #initializae parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    #Iterate over each variable on the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        #Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:] #skip the 'model/' prefix
 
        #Identity the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        #Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key,{})

        #Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
