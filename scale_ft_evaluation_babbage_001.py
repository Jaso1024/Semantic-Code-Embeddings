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
                yield [f'{anchor} <SEP> {augmentation}' for anchor, augmentation in zip(anchor_batch, augmentation_batch)], labels_batch
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
                yield [f'{anchor} <SEP> {augmentation}' for anchor, augmentation in zip(anchor_batch, augmentation_batch)], labels_batch
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

"""# Salesforce CodeGen-350M-Mono Test

### Modules

### Training
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
import openai
import json
import time

MAX_RETRIES = 5

def evaluate_accuracy(dataset_eval, save_file="progress.json"):
    all_predictions = []
    all_labels = []

    try:
        with open(save_file, "r") as f:
            progress_data = json.load(f)
            start_idx = progress_data["index"]
            all_predictions = progress_data["predictions"]
            all_labels = progress_data["labels"]
    except (FileNotFoundError, json.JSONDecodeError):
        start_idx = 0

    for idx, datapoint in enumerate(list(dataset_eval)[start_idx:], start=start_idx):
        datapoint_predictions = []

        for point in datapoint[0]:
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    response = openai.Completion.create(
                      model="text-babbage-001",
                      prompt=f"Are the following pieces of code equivalent? \n {point}  \n These pieces of code",
                      temperature=0
                    )

                    response_content = response.choices[0]['text'].lower().replace('\n', '')

                    if 'no' in response_content or 'are not equivalent' in response_content:
                        prediction = 0
                    elif 'yes' in response_content or 'are equivalent' in response_content or 'produce the same result' in response_content:
                        prediction = 1
                    else:
                        retries += 1
                        print(response)
                        continue  # Retry

                    datapoint_predictions.append(prediction)
                    break  # Valid prediction, exit loop
                except openai.error.OpenAIError:
                    retries += 1
                    time.sleep(5)
                    print(response)
                    print(f"An error occurred while processing datapoint {idx}. Retrying...")
                    continue

            if retries == MAX_RETRIES:
                raise RuntimeError(f"Failed to make valid prediction after {MAX_RETRIES} retries for datapoint {idx}")

        all_predictions.extend(datapoint_predictions)
        all_labels.extend(datapoint[1])

        progress_data = {
            "index": idx + 1,
            "predictions": all_predictions,
            "labels": all_labels
        }
        with open(save_file, "w") as f:
            json.dump(progress_data, f)


    return all_predictions, all_labels

preds, labels = evaluate_accuracy(contrastive_data_generator_eval(val_probs, batch_size=1))

with open("progress.json", "r") as f:
            progress_data = json.load(f)
            start_idx = progress_data["index"]
            preds = progress_data["predictions"]
            labels = progress_data["labels"]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)
f1 = f1_score(labels, preds)

print(len(labels))

print(accuracy, precision, recall, f1)