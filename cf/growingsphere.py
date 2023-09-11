#Filename:	growsphere.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 13 Apr 2022 08:34:19 

import torch
import numpy as np
import copy

class GrowingSphere(object):

    def __init__(self, target, model):
        
        self.target = target
        self.model = model

    def generate_counterfactual(self, input_x, eta, observation_n = 30):
        """
        Generate counterfactual explanations from the sphere around an input instance.
        Arguments:
            input_x: input instance.
            eta: initial radius.
        Returns:
            a counterfactual explanation.
        """
        eta, zs = self.find_eta(eta, observation_n, input_x)
        num, cfs = self.find_cf(eta, 2 * eta, zs, observation_n, input_x)
        if num == 1:
            cf_star = self.find_sparse_cf(cfs, input_x)
        else:
            cf_star = np.zeros_like(cfs)
            for i in range(len(cfs)):
                cf_star[i] = self.find_sparse_cf(cfs[i], input_x)

        return cf_star

    def check_if_in_SL(self, z, low, high, input_x):
        """
        check a instance whether in the sphere of radius between low and high.
        Arguments:
            z: a sampled instance.
            low: the smallest radius.
            high: the largest radius.
            input_x: an input instance.
        Return:
            False, True
        """
        norm_val = np.linalg.norm(input_x - z)
        if norm_val >= low and norm_val <= high:
            return True
        else:
            return False
    
    def sample(self, eta, low, high, input_x):
        """
        return a single sample.
        Arguments:
            low:
            high:
            observation_n: number of sampled observations.
        """
        random_vector = np.random.uniform(-1, 1, input_x.shape[1]).astype(np.float32)
        a = eta / np.sqrt(np.sum(np.square(random_vector)))
        b = a * random_vector
        random_vector = b + input_x

        if self.check_if_in_SL(random_vector, low, high, input_x):
            return random_vector
        else:
            return self.sample(eta, low, high, input_x)

    def make_z(self, eta, low, high, input_x, observation_n):
        """
        Sample n observations from the sphere.
        Arguments:
            eta: radius.
            low:
            high:
            input_x: an input instance.
            observation_n: number of sampled observations.
        """
        zs = np.zeros((observation_n, input_x.shape[1])).astype(np.float32)
        for i in range(observation_n):
            zs[i] = self.sample(eta, low, high, input_x)

        return zs

    def binary_eta(self, z, eta):
        """
        reduce eta by 1/2
        Arguments:
            z: samples.
            eta: radius.
        """
        if (self.model.predict(z) == self.target).any():
            return eta/2
        else:
            return None
    
    def find_eta(self, radius_eta, observation_n, input_x):
        """
        Find the minimal eta.
        Arguments:
            radius_eta: the input radius.
            observation_n: number of samples.
            input_x: an input instance.
        """
        eta = radius_eta
        zs = self.make_z(eta, 0, eta, input_x, observation_n)
        tmp = self.binary_eta(zs, eta)
        while tmp is not None:
            eta = tmp 
            zs = self.make_z(tmp, 0, tmp, input_x, observation_n)
            tmp = self.binary_eta(zs, tmp)

        return eta, zs

    def find_cf(self, low, high, zs, observation_n, input_x):
        """
        Sample n observations from the sphere.
        Arguments:
            eta: radius.
            low: lower radius.
            high: high radius.
            zs: a set of samples.
            input_x: an input instance.
            observation_n: number of sampled observations.
        """

        eta = low
        while True:
            if not (self.model.predict(zs) == self.target).any():
                zs = self.make_z(low, low, high, input_x, observation_n)
                low = high
                high = high + eta
            else:
                break
        prediction = self.model.predict(zs)
        idx = np.argwhere(prediction == self.target).squeeze()
        num = np.sum(prediction == self.target)
        return num, zs[idx]

    
    def find_sparse_cf(self, cfs, input_x):
        """
        post-processing for sparsity.
        Arguments:
            cfs: a set of counterfactual explanations. d dimension.
            input_x: an input instance. 1 * d
        """
        cfs_prime = cfs.copy()
        non_zero_indices = np.argwhere(cfs_prime != input_x[0]).squeeze().tolist()

        while len(non_zero_indices) > 0:
            argmin = np.argmin(np.abs(cfs_prime[non_zero_indices] - input_x[0, non_zero_indices]))
            cfs_prime[non_zero_indices[argmin]] = input_x[0, non_zero_indices[argmin]]
            if self.model.predict(cfs_prime[np.newaxis, :]) != self.target:
                cfs_prime[non_zero_indices[argmin]] = cfs[non_zero_indices[argmin]]

            non_zero_indices = np.delete(non_zero_indices, argmin)

        return cfs_prime
                

