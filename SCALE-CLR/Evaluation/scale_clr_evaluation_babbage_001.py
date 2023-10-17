# -*- coding: utf-8 -*-


import os
import numpy as np
import re
import requests
import tarfile
import shutil
import pandas as pd

"""## Data Loading"""

import pickle
with open('CodeNetPythonTest', 'rb') as file:
  python_list = pickle.load(file)

"""## Data Generators"""

from itertools import combinations
import random
import os
import numpy as np
import re
import requests
import tarfile
import shutil
import pandas as pd

import random

def contrastive_data_generator_train(data, batch_size=2, max_negatives=1, max_sequence_length=1500):
    data_length = len(data)
    random.seed(None)

    def select_anchor(row_index):
          while True:
              anchor_index = random.choice(list(range(len(data[row_index]))))
              anchor = data[row_index][anchor_index]
              if len(anchor) <= max_sequence_length:
                  return anchor_index, anchor

    def select_positive(row_index, anchor_index):
          while True:
              positive_index = anchor_index
              while positive_index == anchor_index:
                  positive_index = random.choice(list(range(len(data[row_index]))))
              positive = data[row_index][positive_index]
              if len(positive) <= max_sequence_length:
                  return positive

    def select_negative(anchor_row_index):
          while True:
              negative_row_index = anchor_row_index
              while negative_row_index == anchor_row_index:
                  negative_row_index = random.choice(list(range(data_length)))
              negative = random.choice(data[negative_row_index])
              if len(negative) <= max_sequence_length:
                  return negative


    def datapoint_generator():

            anchor_batch = []
            augmentation_batch = []
            labels_batch = []

            for row_index in range(data_length):
              random.seed(None)
              if len(anchor_batch) >= batch_size:
                yield anchor_batch, augmentation_batch, labels_batch
                anchor_batch = []
                augmentation_batch = []
                labels_batch = []
              anchor_index, anchor = select_anchor(row_index)
              positive = select_positive(row_index, anchor_index)
              anchor_batch.append(anchor)
              augmentation_batch.append(positive)
              labels_batch.append(1)






    return datapoint_generator()

import random

def contrastive_data_generator_eval(data, batch_size=1, max_negatives=1, max_sequence_length=1500, seed=42):
    random.seed(seed)

    data = [[s for s in inner_list if len(s) <= max_sequence_length] for inner_list in data]

    data = [inner_list for inner_list in data if len(inner_list) >= 2]

    data_length = len(data)

    def select_anchor(row_index):
          while True:
              anchor_index = random.choice(list(range(len(data[row_index]))))
              anchor = data[row_index][anchor_index]
              if len(anchor) <= max_sequence_length:
                  return anchor_index, anchor

    def select_positive(row_index, anchor_index):
          while True:
              positive_index = anchor_index
              while positive_index == anchor_index:
                  positive_index = random.choice(list(range(len(data[row_index]))))
              positive = data[row_index][positive_index]
              if len(positive) <= max_sequence_length:
                  return positive

    def select_negative(anchor_row_index):
          while True:
              negative_row_index = anchor_row_index
              while negative_row_index == anchor_row_index:
                  negative_row_index = random.choice(list(range(data_length)))
              negative = random.choice(data[negative_row_index])
              if len(negative) <= max_sequence_length:
                  return negative


    def datapoint_generator():

            anchor_batch = []
            augmentation_batch = []
            labels_batch = []

            for row_index in range(data_length):
              if len(anchor_batch) >= batch_size:
                yield anchor_batch, augmentation_batch, labels_batch
                anchor_batch = []
                augmentation_batch = []
                labels_batch = []
              anchor_index, anchor = select_anchor(row_index)
              positive = select_positive(row_index, anchor_index)
              negative1 = select_negative(row_index)
              negative2 = select_negative(row_index)
              negative3 = select_negative(row_index)
              anchor_batch.extend([anchor, anchor, anchor, anchor])
              augmentation_batch.extend([positive, negative1, negative2, negative3])
              labels_batch.extend([1, 0, 0, 0])






    return datapoint_generator()

for _, point in zip(range(1), contrastive_data_generator_train(python_list, batch_size=4)):
  anchors, augmentations, labels = point
  print(anchors)
  print('---------------------')
  print(augmentations)
  print('---------------------')
  print(labels)

"""# SCALE CLR Eval

### Modules
"""

val_probs = python_list

"""-----"""


import openai
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

openai.api_key = ''

from sklearn.metrics import accuracy_score
import torch
import openai
import pickle
import os

def evaluate_accuracy(dataset_eval, threshold=0.8):
    all_predictions = []
    all_labels = []

    saved_files = [filename for filename in os.listdir() if filename.startswith('predictions_labels_') and filename.endswith('.pkl')]
    if saved_files:
        latest_saved_file = max(saved_files)
        latest_idx = int(latest_saved_file.split('_')[2].split('.')[0])
        dataset_eval = list(dataset_eval)[latest_idx + 1:]  # Start from the next datapoint

    for idx, datapoint in enumerate(dataset_eval):
        code1_embeddings = [
            openai.Embedding.create(
                input=datapoint[0][0],
                model="code-search-babbage-code-001"
            )['data'][0]['embedding'],
            openai.Embedding.create(
                input=datapoint[0][1],
                model="code-search-babbage-code-001"
            )['data'][0]['embedding'],
            openai.Embedding.create(
                input=datapoint[0][2],
                model="code-search-babbage-code-001"
            )['data'][0]['embedding'],
            openai.Embedding.create(
                input=datapoint[0][3],
                model="code-search-babbage-code-001"
            )['data'][0]['embedding'],
        ]

        code2_embeddings = [
            openai.Embedding.create(
                input=datapoint[1][0],
                model="code-search-babbage-code-001"
            )['data'][0]['embedding'],
            openai.Embedding.create(
                input=datapoint[1][1],
                model="code-search-babbage-code-001"
            )['data'][0]['embedding'],
            openai.Embedding.create(
                input=datapoint[1][2],
                model="code-search-babbage-code-001"
            )['data'][0]['embedding'],
            openai.Embedding.create(
                input=datapoint[1][3],
                model="code-search-babbage-code-001"
            )['data'][0]['embedding'],
        ]

        similarity = F.cosine_similarity(torch.tensor(code1_embeddings), torch.tensor(code2_embeddings), dim=-1)
        predictions = (similarity > threshold).float()
        all_predictions.extend(predictions.tolist())
        all_labels.extend(datapoint[2])

        # Save predictions and labels at every datapoint using pickle
        save_filename = f'predictions_labels_{idx}.pkl'
        with open(save_filename, 'wb') as file:
            pickle.dump((predictions.tolist(), datapoint[2]), file)

    return all_predictions, all_labels



preds, labels = evaluate_accuracy(contrastive_data_generator_eval(val_probs, batch_size=1))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)
f1 = f1_score(labels, preds)

print(accuracy, precision, recall, f1)