# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:03:25 2019

@author: carstenschelp
"""
import numpy as np

class MyDumbClusterer():
  
  def fit(self, X):
    self.labels_ = np.zeros((X.shape[0], ))