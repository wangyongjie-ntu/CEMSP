#Filename:	plaincf.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 13 Apr 2022 08:34:11 

import torch
import torch.nn.functional as F

class PlainCF(object):
    def __init__(self, target, model):
        """
        Initilization.
            target: target class.
            model: the pretrain model interface.
        """
        self.target = target
        self.model = model

    def generate_counterfactuals(self, input_x, _lambda, optimizer, lr, min_iter = 500, max_iter = 5000, loss_diff_thres = 1e-4):
        """
        generate counterfactual explanations for a single input.
        Arguments:
           input_x: an input instance.
           _lambda: trade-off factor between cost and validity.
           optimizer: optimizer for minimizing the lagrange term.
           lr: learning rate.
           max_iter: maximum iteration.
        """

        input_x = torch.FloatTensor(input_x)
        self._lambda = _lambda
        inital_point = torch.randn(input_x.shape)
        inital_point = torch.FloatTensor(inital_point)
        inital_point.requires_grad_(True)
        
        if optimizer == 'adam':
            optim = torch.optim.Adam([inital_point], lr)
        else:
            optim = torch.optim.RMSprop([inital_point], lr)

        loss_diff = 0
        iteration = 0
        self.loss_converge_iter = 0
        self.loss_converge_max_iter = 2
        cur_loss = torch.Tensor([0.])

        while self.stop_loop(inital_point, iteration, min_iter, max_iter, loss_diff, loss_diff_thres):
            optim.zero_grad()
            pre_loss = self.total_loss(inital_point, input_x)
            pre_loss.backward()
            #denominator = torch.linalg.norm(inital_point.grad, dim = 1) + 1e-6
            #denominator = denominator.repeat(inital_point.shape[1]).reshape(inital_point.shape[1], inital_point.shape[0]).T
            #inital_point.grad = inital_point.grad / denominator
            optim.step()
            loss_diff = abs(cur_loss - pre_loss)
            cur_loss = pre_loss
            iteration += 1
        
        return inital_point.detach().numpy()

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

    def yloss_dice(self, probs):
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

    def yloss(self, probs):

        loss = F.relu(self.target - probs)
        return loss

    def total_loss(self, inital_point, input_x):
        """
        compute the total loss.
        Arguments:
            inital_point: a data point.
            input_x: an input instance.

        return: total loss
        """
        probs = self.model.predict_tensor(inital_point)[1]
        yloss = self.yloss(probs)
        loss2 = torch.mean(torch.abs(inital_point - input_x))
        return self._lambda * yloss + loss2

