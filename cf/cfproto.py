#Filename:	dice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 13 Apr 2022 08:34:22 

import numpy as np
import torch.nn.functional as F
import copy
import torch

class CFProto(object):
    """
    This algorithm corresponds to the problem definition in Eq(15) C of paper "Interpretable Counterfactual Explanations Guided by Prototypes", ECML 2020
    The hyper-parameter of beta is the same as it in Alibi. The hyper-parameter of c for attack is fixed in our paper. 
    """

    def __init__(self, target, model):
        """
        Initialization.
            target: target class.
            model: pretrained model interface.
        """
        self.target = target
        self.target_class = round(target)
        self.model = model

    def generate_counterfactuals(self, input_x, proto, proximity_weight = 5, l1_weight = 0.1, 
            optimizer = "adam", lr = 0.001, min_iter= 500, max_iter = 5000, loss_diff_thres = 1e-4):

        """
        Generate diverse counterfactual explanations.
        Arguments:
            input_x: an input instance.
            inital_points: inital points for search process.
            proto: prototype of target class.
            proximity_weight: trade-off factor.
            l1_weight: trade-off factor.
            optimizer: optimizer for minimizing.
            lr: learning rate
            min_iter: minimum iterations.
            max_iter: maximum iterations.
            loss_diff_thres: the loss threshold between two consecutive iterations.
        """

        input_x = torch.from_numpy(input_x)
        proto = torch.from_numpy(proto)
        inital_points = torch.randn(input_x.shape)
        inital_points.requires_grad_(True)

        if optimizer == "adam":
            optimizer = torch.optim.Adam([inital_points], lr = lr)
        else:
            optimizer = torch.optim.RMSprop([inital_points], lr = lr)

        loss_diff = 0
        iteration = 0
        self.loss_converge_iter = 0
        self.loss_converge_max_iter = 2
        cur_loss = torch.Tensor([0.])
        while self.stop_loop(inital_points, iteration, min_iter, max_iter, loss_diff, loss_diff_thres):
            optimizer.zero_grad()
            pre_loss = self.total_loss(inital_points, input_x, proto, l1_weight, proximity_weight)
            pre_loss.backward()
            denominator = torch.linalg.norm(inital_points.grad, dim = 1) + 1e-6
            denominator = denominator.repeat(inital_points.shape[1]).reshape(inital_points.shape[1], inital_points.shape[0]).T
            inital_points.grad = inital_points.grad / denominator
            optimizer.step()
            loss_diff = abs(cur_loss - pre_loss)
            cur_loss = pre_loss
            iteration += 1
        return inital_points.detach().numpy()

    def stop_loop(self, cfs, iteration, min_iter, max_iter, loss_diff, loss_diff_thres):
        """
        Stop conditions.
        Arguments:
            iteration: current iteration number.
            min_iter: minimum iteration number.
            max_iter: maximum iteration number.
            loss_diff: the diffference of loss.
            loss_diff_thres: the preset threshold for loss.
        """

        if iteration < min_iter:
            return True

        if iteration > max_iter:
            return False

        test_preds = self.model.predict_tensor(cfs)[1]
        if (test_preds >= self.target).all():
            if loss_diff < loss_diff_thres:
                self.loss_converge_iter += 1
                if self.loss_converge_iter < self.loss_converge_max_iter:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return True

    def yloss(self, probs):
        """
        Computes the first part hinge loss (y-loss) of the loss function.
        Arguments:
            probs: probabilities of a set of counterfactual explanations.
        """

        loss = F.relu(self.target - probs)
        return loss
    
    def l1_loss(self, cfs, input_x):
        """
        compute the Euclidean / L1 distance between cfs and input_x.
        Arguments:
            cfs: a set of counterfactual explanations.
            input_x: an input instance.
        """
        return (cfs - input_x).abs().mean()

    def l2_loss(self, cfs, input_x):
        """
        compute the Euclidean / L1 distance between cfs and input_x.
        Arguments:
            cfs: a set of counterfactual explanations.
            input_x: an input instance.
        """
        return (cfs - input_x).pow(2).mean()

    def protoloss(self, cfs, proto):
        """
        compute the proto loss between cfs and a specified prototype.
        Argugemts:
            cfs: counterfactual explanations.
        """
        return (cfs - proto).pow(2).mean()

    def total_loss(self, cfs, input_x, proto, l1_weight, proximity_weight):
        """
        The total loss.
        Arguments:
            cfs: a set of counterfactual explanations.
            input_x: an input instance.
            proto: prototype of target class.
            l1_weight: trade-off factor
            proximity_weight: trade-off factor.
        """
        probs = self.model.predict_tensor(cfs)[1]
        yloss = self.yloss(probs)
        l1_loss = self.l1_loss(cfs, input_x)
        l2_loss = self.l2_loss(cfs, input_x)
        proto_loss = self.protoloss(cfs, proto)
        total_loss = proximity_weight * yloss + l1_weight * l1_loss + l2_loss + proto_loss

        return total_loss

