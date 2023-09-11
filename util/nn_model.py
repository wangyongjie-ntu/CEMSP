#Filename:	nn_model.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 11 Apr 2022 05:10:14 

import torch
import os
import numpy as np
from util.base_model import BaseModel
import torch.nn.functional as F

class NNModel(BaseModel):

    def __init__(self, model_path):
        """
        Initialization function.
        Arguments:
            model_path: path of pretrained model.
        """
        
        self.model_path = model_path
        self.load_model()
        self.model = self.model.cpu() # compute on CPU device
        self.model.eval() # work on eval mode

    def load_model(self):

        if os.path.exists(self.model_path):
            self.model = torch.load(self.model_path)
        else:
            self.model = None

    def predict(self, input_x):
        """
        Arguments:
            input_x: input numpy array
        Returns: prediction
        """
        with torch.no_grad():
            _input = torch.from_numpy(input_x)
            pred = self.model(_input)
            _, output = torch.max(pred, 1)
        
        return output.squeeze().numpy()
    
    def predict_tensor(self, input_tensor):
        """
        Arguments:
            input_tensor: input tensor
        Returns:
            prediction.
        """
        pred = self.model(input_tensor)
        prob = F.softmax(pred, dim = 1)
        _, output = torch.max(pred, 1)
        return output, prob[:, 1]
