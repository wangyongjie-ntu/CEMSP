#Filename:	evaluator.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 13 Apr 2022 10:28:44 

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean

class Evaluator(object):
    """ A evaluator on a certain dataset"""
    def __init__(self, dataset, tp_set):
        """
        Initializaition.
        Arguments:
            dataset: train set.
            tp_set: true positive dataset.
        """

        self.dataset = dataset
        self.true_positive = tp_set
        self.mads = None

    def sparsity(self, input_x, cfs, precision = 3):
        """
        compute the sparsity between an input instance and its counterfactual explanation set.
        Arguments:
            input_x: an input instance.
            cfs: a set of counterfactual explanations of input_x.
            precision: the tolerance of measuring the difference between two float variable.
        returns: 
            sparsity.
        """
        lens = len(cfs)
        sparsity = 0
        for i in range(lens):
            tmp = (input_x - cfs[i]).round(precision)
            sparsity += (tmp == 0).sum()

        sparsity = sparsity / (lens * len(cfs[0]))
        return sparsity

    def average_percentile_shift(self, input_x, cfs):
        """
        compute the average percentile shift between an input instance and its counterfactual explanation set.
        Arguments:
            input_x: an input instance.
            cfs: a set of counterfactual explanations of input_x.
        returns: 
            average percentile shift (APS).
        """
        lens = len(cfs)
        shift = np.zeros(input_x.shape[1])
        for i in range(lens):
            for j in range(input_x.shape[1]):
                src_percentile = stats.percentileofscore(self.dataset[:, j], input_x[:, j])
                tgt_percentile = stats.percentileofscore(self.dataset[:, j], cfs[i, j])
                shift[j] += abs(src_percentile - tgt_percentile)

        APS = shift.sum() / (100 * input_x.shape[1])
        return APS
    
    def HausdorffScore(self, cfs1, cfs2):
        """
        compute the modified Hausdorff score for measuring the consistency between two sets of counterfactual explanations.
        Arguments:
            cfs1: the first set of counterfactual explanations.
            cfs2: the second set of counterfactual explanations.
        returns: 
            modified Hausdorff distance between two sets.
        """
        cfs1, cfs2 = np.array(cfs1), np.array(cfs2)
        pairwise_distance = cdist(cfs1, cfs2)
        h_A_B = pairwise_distance.min(1).mean()
        h_B_A = pairwise_distance.min(0).mean()
        return max(h_A_B, h_B_A)
    
    def proximity(self, cfs):
        """
        compute the proximity between an input instance and its counterfacutal explanation set.
        Arguments:
            cfs: a set of counterfactual explanations of an instance.
        returns: 
            proximity score.
        """
        lens = len(cfs)
        proximity = 0
        print(cfs)
        for i in range(lens):
            cf = cfs[i:i+1]
            distance = cdist(cf, self.true_positive).squeeze()
            _min = np.argmin(distance)
            povit = self.true_positive[_min]
            povit_ = povit[np.newaxis, :]
            distance1 = cdist(povit_, self.true_positive).squeeze()
            distance1[_min] = float('inf') # disable the distance between povit and povit
            _min1 = np.argmin(distance1)
            povit1 = self.true_positive[_min1]
            # print(cf.shape, povit.shape, povit1.shape)
            # 把cf -> cf.squeeze()
            proximity += euclidean(cf.squeeze(), povit) / (euclidean(povit, povit1) + 1e-6)
        
        return proximity / lens

    def count_diversity(self, cfs, precision=3):
        """
        compute thr count-diversity between counterfactual explanation set.
        Args:
            cfs: a set of counterfactual explanations of an instance.

        Returns:
            diversity score
        """
        k, d = cfs.shape
        if k <= 1:
            return -1
        diversity = 0
        for i in range(k-1):
            for j in range(i+1, k):
                tmp = (cfs[i] - cfs[j]).round(precision)
                num_of_differ_features = (tmp != 0).sum()
                diversity += num_of_differ_features
        return diversity * 2 / (k * (k - 1) * d)

    def diversity(self, cfs):
        """
        compute thr count-diversity between counterfactual explanation set.
        Args:
            cfs: a set of counterfactual explanations of an instance.

        Returns:
            diversity score
        """
        k, d = cfs.shape
        if k <= 1:
            return -1
        diversity = 0
        for i in range(k-1):
            for j in range(i+1, k):
                diversity += self.compute_dist(cfs[i], cfs[j])
        return diversity * 2 / (k * (k - 1))

    def get_mads(self):
        """Computes Median Absolute Deviation of features."""
        if self.mads is None:
            self.mads = np.median(
                abs(self.dataset - np.median(self.dataset, axis=0)), axis=0)
        return self.mads

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        # return torch.sum(torch.mul((torch.abs(x_hat - x1)), self.feature_weights_list), dim=0)
        return np.sum(np.multiply(np.abs(x_hat - x1), self.get_mads()), axis=0)

