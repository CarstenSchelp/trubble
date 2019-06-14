# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:03:25 2019

@author: carstenschelp
"""
import numpy as np
import sys
sys.path.append('../src')
import autoboundary as abd
import normpca as npca

class MyDumbClusterer():
  
    def fit(self, X):
        max_depth = 2
        labels = np.repeat(0, X.shape[0])
        depth = 0
        for depth in range(0, max_depth):
            # TODO: EEEuh - how to get all those sub-labels together
            # in a consistent way, again ...
            for label in set(labels):
                ix_label = (labels == label)
                subX = X[ix_label]
                subLabels = self._divide_and(subX, depth)
                labels[ix_label] = 
            
        self.labels_ = labels
    
    def _divide_and(self, X, labelBase):
        nrmpca = npca.NormPCA(X)
        return abd.labelsplit(nrmpca.projection[:, 0])
        
    
#mdc = MyDumbClusterer()
#mdc.fit(np.array([[1,2,2.5,4,5], [3,5,7,9,10]]).T)
#print(mdc.labels_)

