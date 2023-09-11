#Filename:	dt_model.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 13 Agu 2022 03:24:47 

import os
import numpy as np
import pickle
from util.base_model import BaseModel

class DTModel(BaseModel):

    def __init__(self, model_path):
        """
        Initialization function.
        Arguments:
            model_path: path of pretrained model.
        """
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = pickle.load(open(self.model_path, 'rb'))
        else:
            self.model = None

    def predict(self, input_x):
        """
        Arguments:
            input_x: input numpy array
        Returns: prediction
        """
        return self.model.predict(input_x)
