#Filename:	cemsp.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 04 Apr 2022 09:14:42 

import numpy as np
from z3 import *

class MapSolver(object):
    def __init__(self, n):
        """
        Initialization.
        Args:
        n: The number of constraints to map.
        """
        self.solver = Solver()
        self.n = n
        self.all_n = set(range(n))  # used in complement fairly frequently

    def next_seed(self):
        """
        Get the seed from the current model, if there is one.
        Returns:A seed as an array of 0-based constraint indexes.
        """
        if self.solver.check() == unsat:
            return None
        seed = self.all_n.copy()  # default to all True for "high bias"
        model = self.solver.model()
        for x in model:
            if is_false(model[x]):
                seed.remove(int(x.name()))

        return list(seed)

    def complement(self, aset):
        """Return the complement of a given set w.r.t. the set of mapped constraints."""
        return self.all_n.difference(aset)

    def pruneSuperSet(self, frompoint):
        """Block up from a given set."""
        self.solver.add(Or([Not(Bool(str(i))) for i in frompoint]))

    def pruneSubSet(self, frompoint):
        """Block down from a given set."""
        comp = self.complement(frompoint)
        self.solver.add(Or([Bool(str(i)) for i in comp]))

class CFSolver(object):
    def __init__(self, n, model, input_x, to_replace, desired_pred):
        """
        Initialization. Args:
        n: the number of features
        model: the pretrained model
        input_x: an instance to explain
        to_replace: the values to replace with
        desired_pred: the desired prediction
        """
        self.n = n
        self.model = model
        self.input_x = input_x
        self.to_replace = to_replace
        self.desired_pred = desired_pred

    def check_CF(self, mask):
        """
        mask: binary mask vector.
        """

        _input = self.mask2CF(mask)
        output = self.model.predict(_input)
        return (output == self.desired_pred)
        
    def set2array(self, seed):
        """Return the binary numpy array from a give set. """
        mask = np.zeros(self.n).astype(np.float32)
        mask[list(seed)] = 1
        return mask

    def array2set(self, mask):
        """ return the subset of features to replace. """
        return set(np.where(mask == 1)[0])

    def mask2CF(self, mask):
        """ return a feature vector after replacement. """
        return self.input_x * (1 - mask) + self.to_replace * mask

    def complement(self, aset):
        """return complement set"""
        return set(range(self.n)).difference(aset)

    def shrink(self, seed):
        """ shrink operation. find the minimal satisfiable counterfactual explanation."""
        current = set(seed)
        for i in seed:
            if i not in current:
                continue
            current.remove(i)
            if self.check_CF(self.set2array(current)):
                continue
            else:
                current.add(i)

        return current

    def grow(self, seed):
        """ grow operation. find the maximal unsatisfiable subsets."""
        current = seed
        for i in self.complement(current):
            current.append(i)
            if self.check_CF(self.set2array(current)):
                current.pop()

        return current


def FindCF(cfsolver, mapsolver):
    while True:
        seed = mapsolver.next_seed()
        if seed is None:
            return
        mask = cfsolver.set2array(seed)
        if not cfsolver.check_CF(mask):
            A_hat = cfsolver.grow(seed)
            mapsolver.pruneSubSet(A_hat)
        else:
            A_star = cfsolver.shrink(seed)
            mask = cfsolver.set2array(A_star)
            CF = cfsolver.mask2CF(mask)
            yield ("Counterfactual Explanation", CF, mask)
            mapsolver.pruneSuperSet(A_star)

