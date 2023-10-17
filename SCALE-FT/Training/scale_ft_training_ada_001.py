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

import pickle
with open('CodeNetPythonTrain', 'rb') as file:
  train_data = pickle.load(file)

"""## Data Generators"""

print(len(train_data))

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
          attempts = 0
          while True:
              positive_index = anchor_index
              while positive_index == anchor_index:
                  positive_index = random.choice(list(range(len(data[row_index]))))
                  attempts += 1
                  if attempts > 10:
                    assert 0 == 1, "got stuck in selecting positive"
              positive = data[row_index][positive_index]
              if len(positive) <= max_sequence_length:
                  return positive

    def select_negative(anchor_row_index):
          attempts= 0
          while True:
              negative_row_index = anchor_row_index
              while negative_row_index == anchor_row_index:
                  negative_row_index = random.choice(list(range(data_length)))
                  if attempts > 10:
                    assert 0 == 1, "got stuck in selecting negative"
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
              negative1 = select_negative(row_index)
              negative2 = select_negative(row_index)
              negative3 = select_negative(row_index)
              anchor_batch.extend([anchor, anchor, anchor, anchor])
              augmentation_batch.extend([positive, negative1, negative2, negative3])
              labels_batch.extend([1, 0, 0, 0])






    return datapoint_generator()

print(len(list(contrastive_data_generator_train(train_data))))

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


val_probs = python_list

"""-----"""


import openai
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

openai.api_key = ''


import jsonlines

dataset_train = contrastive_data_generator_train(train_data)
with jsonlines.open('CodeNetPythonTrainData.jsonl', mode='w') as writer:
    for datapoint in dataset_train:
        code_snippets, label = datapoint
        for snippet, v, in zip(code_snippets, label):
          completion = None
          if v == 0:
            completion = 'not equivalent'
          elif v == 1:
            completion = 'equivalent'
          if completion == None:
            print('error', v)
          writer.write({"prompt": f"Are the following pieces of code equivalent? \n {snippet}  \n These pieces of code are ", "completion": f"{completion}"})

"""Query openai api for fine-tuning in terminal

# Evaluation
"""

from sklearn.metrics import accuracy_score
import openai
import json
import time

MAX_RETRIES = 1

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
                      model='ada:ft-<>',
                      prompt=f"{point} \n These pieces of code",
                      temperature=0,
                    )
                    print(response)
                    response_content = response.choices[0]['text'].lower().replace('\n', '')
                    print(response_content)
                    if 'not equivalent' in response_content or 'not  equivalent' in response_content:
                        prediction = 0
                    elif 'are equivalent' in response_content or 'are  equivalent' in response_content:
                        prediction = 1
                    else:
                        retries += 1
                        print(response_content)
                        time.sleep(5)
                        continue  # Retry

                    datapoint_predictions.append(prediction)
                    break  # Valid prediction, exit loop
                except openai.error.OpenAIError:
                    print(response)
                    retries += 1
                    time.sleep(5)
                    print(f"An error occurred while processing datapoint {idx}. Retrying...")
                    continue

            if retries == MAX_RETRIES:
                raise RuntimeError(f"Failed to make valid prediction after {MAX_RETRIES} retries for datapoint {idx}")

        all_predictions.extend(datapoint_predictions)
        all_labels.extend(datapoint[1])

        # Save progress after each datapoint
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