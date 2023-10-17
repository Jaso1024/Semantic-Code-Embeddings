# -*- coding: utf-8 -*-

import os
import numpy as np
import re
import requests
import tarfile
import shutil
import pandas as pd

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

def remove_long_strings_and_check_empty(data, max_string_length):
    # Filter out long strings from the inner lists
    filtered_data = [[s for s in inner_list if len(s) <= max_string_length] for inner_list in data]

    # Check if any inner list has a length of 0
    is_any_empty = any(len(inner_list) == 0 for inner_list in filtered_data)

    return filtered_data, is_any_empty

# Example usage:
data = [['abc', 'de', 'fgh'], ['ijkl'], ['mnopqrs', 'tuv'], [], ['wxyz']]
max_string_length = 3

filtered_data, is_any_empty = remove_long_strings_and_check_empty(data, max_string_length)
print("Filtered Data:", filtered_data)
print("Is any inner list empty?", is_any_empty)

import random

def contrastive_data_generator_train(data, batch_size=2, max_negatives=1, max_sequence_length=3000):

    random.seed(None)

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

"""# Salesforce CodeGen-350M-Mono Test

### Libraries
"""

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from tqdm import tqdm
from operator import itemgetter

import json

from pytorch_metric_learning.losses import NTXentLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

"""### Config"""

class config:
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'Salesforce/codegen-350M-mono'
    text_embedding = 1024
    max_length = 500

    pretrained = True
    trainable_language = True
    trainable_logic = True
    temperature = .7

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.01

"""### Modules"""

class Encoder(nn.Module):
    def __init__(self, model, trainable, pretrained=config.pretrained):
        super().__init__()
        self.model = model

        for parameter in self.model.parameters():
            parameter.requires_grad = trainable

        self.target_token_idx = 0

    def change_trainable(self, trainable):
      for parameter in self.model.parameters():
            parameter.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=config.projection_dim,
        dropout=config.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.relu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.relu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class SimCLR(nn.Module):
    def __init__(self, code1_encoder,  temperature=config.temperature, text_embedding=config.text_embedding):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-350M-mono')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #self.tokenizer.padding_side = 'left'

        self.code1_encoder = code1_encoder
        self.code1_projection_head = ProjectionHead(embedding_dim=text_embedding)


        self.temperature = temperature

        self.eps = 1e-7

        self.loss = NTXentLoss(temperature=temperature)

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds))
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def get_loss(self, z_i, z_j):
        embeddings = torch.cat((z_i, z_j))
        indices = torch.arange(0, z_i.size(0), device=z_i.device)
        labels = torch.cat((indices, indices))

        return self.loss(embeddings, labels)



    def forward(self, batch, train=True):
        code1, attention_mask_code1 = itemgetter('input_ids', 'attention_mask')(self.tokenizer.batch_encode_plus(batch['code1'], padding=True))
        code2, attention_mask_code2 = itemgetter('input_ids', 'attention_mask')(self.tokenizer.batch_encode_plus(batch['code2'], padding=True))

        code1 = torch.tensor(code1).to(device)
        attention_mask_code1 = torch.tensor(attention_mask_code1).to(device)

        code2 = torch.tensor(code2).to(device)
        attention_mask_code2 = torch.tensor(attention_mask_code2).to(device)

        code1_embeddings = self.code1_encoder(input_ids=code1, attention_mask=attention_mask_code1)
        code2_embeddings = self.code1_encoder(input_ids=code2, attention_mask=attention_mask_code2)

        if train:
          code1_embeddings = self.code1_projection_head(code1_embeddings)
          code2_embeddings = self.code1_projection_head(code2_embeddings)

        return code1_embeddings, code2_embeddings

"""### Training"""

import pickle

with open('CodeNetPythonTest', 'rb') as file:
  python_list = pickle.load(file)

with open('CodeNetPythonTrain', 'rb') as file:
  train_data = pickle.load(file)

train_probs = train_data[50:]

val_probs = train_data[:50]

eval_probs = python_list

def count_common_elements(list1, list2):
    set1 = set([elem for sublist in list1 for elem in sublist])
    set2 = set([elem for sublist in list2 for elem in sublist])
    common_elements = set1 & set2
    return len(common_elements)

print(count_common_elements(train_probs, python_list))

def remove_common_elements(list1, list2):
    set1 = set([elem for sublist in list1 for elem in sublist])
    set2 = set([elem for sublist in list2 for elem in sublist])
    common_elements = set1 & set2

    new_list1 = []
    for sublist in list1:
        new_sublist = [elem for elem in sublist if elem not in common_elements]
        new_list1.append(new_sublist)

    return new_list1

train_probs = remove_common_elements(train_probs, python_list)

print(count_common_elements(train_probs, python_list))

"""-----"""

code1_encoder = Encoder(AutoModel.from_pretrained(config.model_name), trainable=config.trainable_language).to(device)

contrastive_learner = SimCLR(code1_encoder=code1_encoder).to(device)

contrastive_learner_optimizer = optim.AdamW(contrastive_learner.parameters(), lr=1e-6)

torch.cuda.empty_cache()

from sklearn.metrics import accuracy_score

def evaluate_accuracy(contrastive_learner, dataset_eval, threshold=0.8):
    # Set the model in evaluation mode
    contrastive_learner.eval()

    with torch.no_grad():
        all_predictions = []
        all_labels = []

        for idx, datapoint in enumerate(dataset_eval):
            code1_embeddings, code2_embeddings = contrastive_learner({
                "code1": datapoint[0],
                "code2": datapoint[1]
            }, train=False)

            similarity = F.cosine_similarity(code1_embeddings, code2_embeddings, dim=-1)
            predictions = (similarity > threshold).float()
            all_predictions.extend(predictions.tolist())
            all_labels.extend(datapoint[2])
            # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)

    return accuracy, all_predictions, all_labels

epochs = 20
batch_size = 2
negatives_ratio = 15

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(contrastive_learner_optimizer, T_max=50, eta_min=1e-8)

loss_list = []
accs_list = []
preds_list = []
best_accuracy = 0


dataset_eval = contrastive_data_generator_eval(eval_probs, batch_size=1)
acc, preds, labels = evaluate_accuracy(contrastive_learner, dataset_eval, threshold=.8)
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)
f1 = f1_score(labels, preds)
print('Initial Stats', accuracy, precision, recall, f1)

accs_list.append(acc)
preds_list.append(preds)

for epoch in range(epochs):
    progress_bar = tqdm(range(0, 1000000 , batch_size))

    dataset_val = contrastive_data_generator_eval(val_probs, batch_size=1)
    acc, preds, labels = evaluate_accuracy(contrastive_learner, dataset_val, threshold=.8)
    print(f'Validation Accuracy: {acc}')
    accs_list.append(acc)
    preds_list.append(preds)
    if acc > best_accuracy:
        torch.save(contrastive_learner.state_dict(), "Approach2CodeGen350.pth")
        best_accuracy=acc
    dataset_train = contrastive_data_generator_train(train_probs, batch_size=2)
    for idx, datapoint in zip(progress_bar, dataset_train):
      contrastive_learner_optimizer.zero_grad()
      code1_embeddings, code2_embeddings = contrastive_learner({
            "code1": datapoint[0],
            "code2": datapoint[1]
        })
      loss = contrastive_learner.get_loss(code1_embeddings, code2_embeddings)
      loss_list.append(loss.item())

      loss.backward()
      contrastive_learner_optimizer.step()
      torch.cuda.empty_cache()
      progress_bar.set_description(f'Epoch: {epoch+1} | Running Loss: {sum(loss_list)/len(loss_list)} | Current Loss: {loss_list[-1]}')
      if idx % 500 == 0: # using modulus here causes unexpected behavior
        dataset_val = contrastive_data_generator_eval(val_probs, batch_size=1)
        acc, preds, labels = evaluate_accuracy(contrastive_learner, dataset_val, threshold=.8)
        print(f'Validation Accuracy: {acc}')
        accs_list.append(acc)
        preds_list.append(preds)
        if acc > best_accuracy:
          torch.save(contrastive_learner.state_dict(), "Approach2CodeGen350.pth")
          best_accuracy=acc
        scheduler.step()
        torch.cuda.empty_cache()

print("done training")

import pickle

contrastive_learner.load_state_dict(torch.load("Approach2CodeGen350.pth"))
dataset_eval = contrastive_data_generator_eval(eval_probs, batch_size=1)
acc, preds, labels = evaluate_accuracy(contrastive_learner, dataset_eval, threshold=.8)
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)
f1 = f1_score(labels, preds)
print('loaded weights stats', accuracy, precision, recall, f1)

accs_list.append(acc)
preds_list.append(preds)

output_file = 'preds_list.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(preds_list, f)

print(f'Preds_list saved to {output_file}')

