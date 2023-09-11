#Filename:	base_model.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 11 Apr 2022 05:08:23 

class BaseModel(object):

    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        raise NotImplementedError

    def predict(self, input_x):
        raise NotImplementedError

    def gradient(self, input_x):
        raise NotImplementedError


