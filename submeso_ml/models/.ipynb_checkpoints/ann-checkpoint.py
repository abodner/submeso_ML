import torch
import torch.nn as nn
import numpy as np
import itertools

""" Fully connected artificial neural network """
    
    
class ANN(torch.nn.Module):
    def __init__(self, config):
        super(ANN, self).__init__()

        input_size = config['input_size']
        output_size = config['output_size']
        hidden_size = config['hidden_size']

        self.input_layer = nn.Linear(input_size, hidden_size[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_size) - 1):
            self.hidden_layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
        self.output_layer = nn.Linear(hidden_size[-1], output_size)
        
        self.config = config  # Store the config as an attribute

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x
    
    
    def save_model(self):
        """ Save the model config, and optimised weights and biases. We create a dictionary
        to hold these two sub-dictionaries, and save it as a pickle file """
        if self.config["save_path"] is None:
            print("No save path provided, not saving")
            return
        save_dict={}
        save_dict["state_dict"]=self.state_dict() ## Dict containing optimised weights and biases
        save_dict["config"]=self.config           ## Dict containing config for the dataset and model
        save_string=os.path.join(self.config["save_path"],self.config["save_name"])
        with open(save_string, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved as %s" % save_string)
        return