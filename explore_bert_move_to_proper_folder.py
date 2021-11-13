# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:02:01 2021

@author: 16028
"""

import pandas as pd
import numpy as np
import torch

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = torch.load(r'..\3_explore_bert\data\conll.pt')

train = data['train']
test = data['validation']

train
for key in train:
    print(key)

data['train'][0]['sentence']
len(data['train'][0]['sentence'].split())
len(data['train'][0]['sentence'])

# =============================================================================
# 
# for each sentence:
#     a list of lists 13 tensors (one for each layer)
#     and each tensor has dimension (Length of sentence x embedding dim = 768?)
#     
# the length of the tensor is always at least two more than the length of the sentence
# why? because of CLS and SEP?  
# =============================================================================
#collect all the bert reps for all the words across training set
# training logistic representations
# say X is total number of words across all varaibles on the trianing set
# calculate f1 on the validation set


data['train'][0]['pos_labels']
data['train'][0]['word_token_indices']
#take the mean of the two tokens to get the representation for 8 and 9
data['train'][0]['hidden_states']
# 13 layers of hidden states for each example
data['train'][0]['hidden_states']



hidden_states_array = []
pos_labels_array = []
sentences_array = []
word_token_indices_array = []
ner_labels_array = []
for mydict in train:

    hidden_states_array.append(mydict['hidden_states'])
    pos_labels_array.append(mydict['pos_labels'])
    sentences_array.append(mydict['sentence'])
    word_token_indices_array.append(mydict['word_token_indices'])
    ner_labels_array.append(mydict['ner_labels'])

#words in our sentence to bert tokens

