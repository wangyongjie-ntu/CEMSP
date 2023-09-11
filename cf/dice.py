#Filename:	dice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 13 Apr 2022 08:34:22 

import numpy as np
import torch.nn.functional as F
import copy
import torch

class DiCE(object):

    def __init__(self, target, model):
        """
        Initialization.
            target: target class.
            model: pretrained model interface.
        """
        self.target = target
        self.target_class = round(target)
        self.model = model

    def generate_counterfactuals(self, input_x, inital_points, total_cfs, proximity_weight = 0.5, diversity_weight = 2,
            dloss = "inverse_dist", optimizer = "adam", lr = 0.05, min_iter= 500, max_iter = 5000, loss_diff_thres = 1e-4,
            init_neighbor = True):

        """
        Generate diverse counterfactual explanations.
        Arguments:
            input_x: an input instance.
            inital_points: inital points for search process.
            total_cfs: size of counterfactual explanation set.
            proximity_weight: trade-off factor.
            diversity_weight: trade-off factor.
            y_loss: loss between prediction and desired target.
            dloss: distance used in DPP term.
            optimizer: optimizer for minimizing.
            lr: learning rate
            min_iter: minimum iterations.
            max_iter: maximum iterations.
            loss_diff_thres: the loss threshold between two consecutive iterations.
            init_neighbor: initialization method.
        """

        input_x = torch.from_numpy(input_x)
        #inital_points = input_x.repeat(total_cfs, 1)
        inital_points = torch.from_numpy(inital_points)
        inital_points = inital_points + 0.3 * torch.randn(total_cfs, input_x.shape[1])
        inital_points = torch.FloatTensor(inital_points)
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
            pre_loss = self.total_loss(inital_points, input_x, proximity_weight, diversity_weight)
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
    
        yloss = 0.0
        temp_logits = torch.log(abs(probs - 1e-6) / 
                (1 - abs(probs - 1e-6)))  # 1-e6 for numerical stability.
        criterion = torch.nn.ReLU()
        temp_loss = 1 - temp_logits
        yloss = criterion(temp_loss).mean()
        return yloss
    
    def dloss(self, cfs, input_x):
        """
        compute the Euclidean / L1 distance between cfs and input_x.
        Arguments:
            cfs: a set of counterfactual explanations.
            input_x: an input instance.
        """
        return (cfs - input_x).abs().mean()

    def dpp_loss(self, cfs):
        """
        compute the diversity of a set of counterfactual explanations.
        Arguments:
            cfs: a set of counterfactual explanations.
        """

        if len(cfs.shape) == 1:
            return torch.tensor(0.0)

        det_entries = torch.ones(len(cfs), len(cfs))
        for i in range(len(cfs)):
            for j in range(len(cfs)):
                #det_entries[i, j] = self.dloss(cfs[i], cfs[j])
                det_entries[i, j] = torch.sum((cfs[i] - cfs[j]).abs())

        det_entries = 1.0 / (1.0 + det_entries)
        det_entries += torch.eye(len(cfs)) * 0.0001
        return torch.det(det_entries)

    def total_loss(self, cfs, input_x, proximity_weight, diversity_weight):
        """
        The total loss.
        Arguments:
            cfs: a set of counterfactual explanations.
            input_x: an input instance.
            probs: the probability of target class for cfs.
            yloss_type: methods for computing y loss.
            proximity_weight, diversity_weight: trade-off factor.
        """
        probs = self.model.predict_tensor(cfs)[1]
        yloss = self.yloss(probs)
        dloss = self.dloss(cfs, input_x)
        dpp_loss = self.dpp_loss(cfs)
        total_loss = yloss + proximity_weight * dloss - diversity_weight * dpp_loss
        return total_loss

