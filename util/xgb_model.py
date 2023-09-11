#Filename:	xgb_model.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 14 Jun 2022 11:03:45 

import os
import numpy as np
from xgboost import XGBClassifier
from util.base_model import BaseModel

class XGBModel(BaseModel):

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
            self.model = XGBClassifier()
            self.model.load_model(self.model_path)
        else:
            self.model = None

    def predict(self, input_x):
        """
        Arguments:
            input_x: input numpy array
        Returns: prediction
        """
        return self.model.predict(input_x)

