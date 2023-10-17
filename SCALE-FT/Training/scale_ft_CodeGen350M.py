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

with open("CodeNetPythonTrain", "rb") as fp:
  train_data = pickle.load(fp)
with open("CodeNetPythonTest", "rb") as fp:
  python_list = pickle.load(fp)

"""## Data Generators"""

from peft import LoraModel, LoraConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftModelForFeatureExtraction
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
import random

def contrastive_data_generator_train(data, batch_size=2, max_negatives=1, max_sequence_length=1500):
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
              if len(anchor_batch) >= batch_size:
                yield [f'{anchor} <SEP> {augmentation}' for anchor, augmentation in zip(anchor_batch, augmentation_batch)], labels_batch
                anchor_batch = []
                augmentation_batch = []
                labels_batch = []
              anchor_index, anchor = select_anchor(row_index)
              positive = select_positive(row_index, anchor_index)
              negative1 = select_negative(row_index)
              anchor_batch.extend([anchor, anchor])
              augmentation_batch.extend([positive, negative1])
              labels_batch.extend([1, 0])



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

import bitsandbytes as bnb

from tqdm import tqdm
from operator import itemgetter

from pytorch_metric_learning.losses import NTXentLoss

device = torch.device("cpu")
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

    model_name = 'Salesforce/codegen-350M-mono' #'mrm8488/llama-2-coder-7b'
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
        return last_hidden_state.mean(dim=1)

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
        self.fc = nn.LazyLinear(1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
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

        code1 = torch.tensor(code1)

        code1_embeddings = self.code1_encoder(input_ids=code1, attention_mask=attention_mask_code1)
        return code1_embeddings

"""### Training"""

train_probs = train_data[50:]

val_probs = train_data[:50]

"""-----"""

from transformers import AutoModel

model = AutoModel.from_pretrained(config.model_name)


code1_encoder = Encoder(model, trainable=config.trainable_language)

class ClassifierHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        dropout=config.dropout
    ):
        super().__init__()
        self.sig = nn.LeakyReLU()
        self.fc1 = nn.LazyLinear(1)


    def forward(self, x):
        x = self.fc1(x)
        return x

class Model(nn.Module):
    def __init__(self, encoder, temperature=config.temperature, text_embedding=config.text_embedding):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-350M-mono')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = encoder
        self.classification_head = ClassifierHead(embedding_dim=text_embedding)
    def forward(self, batch):
        input_ids, attention_mask = itemgetter('input_ids', 'attention_mask')(self.tokenizer.batch_encode_plus(batch, padding=True))
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        classification = self.classification_head(embeddings)
        return classification

contrastive_learner = Model(code1_encoder)

contrastive_learner_optimizer = optim.AdamW(contrastive_learner.parameters(), lr=1e-5)

bce_loss = nn.BCEWithLogitsLoss()

torch.cuda.empty_cache()

from sklearn.metrics import accuracy_score

def evaluate_accuracy(contrastive_learner, dataset_eval, threshold=0.5):
    # Set the model in evaluation mode
    contrastive_learner.eval()

    with torch.no_grad():
        all_predictions = []
        all_labels = []

        for idx, datapoint in enumerate(dataset_eval):
            # Get prediction from the model
            code1_prediction = contrastive_learner(
            datapoint[0],
            )

            # Convert the model output to binary predictions based on the given threshold
            binary_predictions = (code1_prediction > threshold).int().cpu().numpy()
            all_predictions.extend(binary_predictions.tolist())
            all_labels.extend(datapoint[1])

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)

    return accuracy

def evaluate_accuracy_(contrastive_learner, dataset_eval, threshold=0.5):
    # Set the model in evaluation mode
    contrastive_learner.eval()

    with torch.no_grad():
        all_predictions = []
        all_labels = []

        for idx, datapoint in enumerate(dataset_eval):
            # Get prediction from the model
            code1_prediction = contrastive_learner(
            datapoint[0],
            )

            # Convert the model output to binary predictions based on the given threshold
            binary_predictions = (code1_prediction > threshold).int().cpu().numpy()
            all_predictions.extend(binary_predictions.tolist())
            all_labels.extend(datapoint[1])

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)

    return accuracy, all_labels, all_predictions



epochs = 20
batch_size = 2
negatives_ratio = 15

best_accuracy = 0
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(contrastive_learner_optimizer, T_max=50, eta_min=1e-6)

dataset_eval = contrastive_data_generator_eval(python_list, batch_size=1)
acc, preds, labels = evaluate_accuracy_(contrastive_learner, dataset_eval, threshold=.5)
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)
f1 = f1_score(labels, preds)
print('Initial stats', accuracy, precision, recall, f1)


loss_list = []
accs_list = []
for epoch in range(epochs):
    progress_bar = tqdm(range(0, 1000000 , batch_size))
    torch.cuda.empty_cache()

    dataset_val = contrastive_data_generator_eval(val_probs, batch_size=1)
    acc = evaluate_accuracy(contrastive_learner, dataset_val, threshold=.5)
    print(f'Validation Accuracy: {acc}')
    if acc > best_accuracy:
          torch.save(contrastive_learner.state_dict(), "Approach1CodeGen350M.pth")
          best_accuracy=acc
    accs_list.append(acc)

    dataset_train = contrastive_data_generator_train(train_probs, batch_size=4, max_sequence_length=1500)
    for idx, datapoint in zip(progress_bar, dataset_train):
      contrastive_learner_optimizer.zero_grad()
      outputs = contrastive_learner(
            datapoint[0],
      )

      targets = torch.tensor(datapoint[1], dtype=torch.float32)
      loss = bce_loss(torch.reshape(outputs, (4,1)), torch.reshape(targets.to(device),(4,1)))
      loss_list.append(loss.item())

      loss.backward()
      contrastive_learner_optimizer.step()
      torch.cuda.empty_cache()
      progress_bar.set_description(f'Epoch: {epoch+1} | Running Loss: {sum(loss_list)/len(loss_list)} | Current Loss: {loss_list[-1]}')
      if idx % 50 == 0: # using modulus here causes unexpected behavior
        dataset_val = contrastive_data_generator_eval(val_probs, batch_size=1)
        acc = evaluate_accuracy(contrastive_learner, dataset_val, threshold=.5)
        print(f'Validation Accuracy: {acc}')
        if acc > best_accuracy:
          torch.save(contrastive_learner.state_dict(), "Approach1CodeGen350M.pth")
          best_accuracy=acc
        accs_list.append(acc)
        scheduler.step()
        torch.cuda.empty_cache()



accuracies = accs_list

import matplotlib.pyplot as plt

def plot_accuracies(accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.show()

plot_accuracies(accuracies)

contrastive_learner.load_state_dict(torch.load("Approach1CodeGen350M.pth"))
dataset_eval = contrastive_data_generator_eval(python_list, batch_size=1)
acc, preds, labels = evaluate_accuracy_(contrastive_learner, dataset_eval, threshold=.5)
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)
f1 = f1_score(labels, preds)
print('loaded weights stats', accuracy, precision, recall, f1)

