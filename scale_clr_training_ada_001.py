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

with open("CodeNetPythonTrain", "rb") as fp:
  train_data = pickle.load(fp)

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

for _, point in zip(range(1), contrastive_data_generator_train(python_list, batch_size=4)):
  anchors, augmentations, labels = point
  print(anchors)
  print('---------------------')
  print(augmentations)
  print('---------------------')
  print(labels)

"""# Salesforce CodeGen-350M-Mono Test

### Modules

### Training
"""

len([point[2] for point in list(contrastive_data_generator_eval(python_list))])

list(contrastive_data_generator_eval(python_list))[1]
len([point[2] for point in list(contrastive_data_generator_eval(python_list))]) * 4

eval_probs = python_list

"""-----"""


import openai
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        out = self.linear2(nn.GELU(self.linear1(x))) + x
        return out

class MLPWithResidual(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(MLPWithResidual, self).__init__()

        self.initial_layer = nn.LazyLinear(hidden_dim)
        self.residual_blocks = nn.ModuleList([nn.LazyLinear(hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.LazyLinear(output_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.gelu(self.initial_layer(x))

        for block in self.residual_blocks:
            out = self.gelu(block(out)) + out

        out = self.output_layer(out)
        return out

input_dim = 10
hidden_dim = 50
output_dim = 2
num_layers = 5

# Create the model
test_model = MLPWithResidual(hidden_dim, output_dim, num_layers)

# Example input tensor
x = torch.randn(64, input_dim)

# Forward pass
output = test_model(x)
print("Output shape:", output.shape)

"""## saving training embeddings"""

import openai
import os
import pickle

# Function to load saved progress
def load_progress(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    return None

# Function to save progress
def save_progress(file_path, progress):
    with open(file_path, 'wb') as file:
        pickle.dump(progress, file)


checkpoint_file = 'progress_checkpoint.pkl'  # Change this to your desired checkpoint filename
checkpoint_interval = 100  # Save progress every 100 iterations

progress = load_progress(checkpoint_file)
if progress is None:
    progress = {'index': 0, 'embeddings': []}

dataset_train = contrastive_data_generator_train(train_data)

try:
    for idx, datapoint in enumerate(dataset_train):
        if idx < progress['index']:
            continue  # Skip already processed data

        code1_embeddings = openai.Embedding.create(
            input=datapoint[0][0],
            model="code-search-ada-code-001"
        )['data'][0]['embedding']

        code2_embeddings = openai.Embedding.create(
            input=datapoint[1][0],
            model="code-search-ada-code-001"
        )['data'][0]['embedding']

        progress['embeddings'].append((code1_embeddings, code2_embeddings))

        if (idx + 1) % checkpoint_interval == 0:
            progress['index'] = idx + 1  # Update the progress index
            save_progress(checkpoint_file, progress)
            print(f"Progress saved at iteration {idx + 1}")

except KeyboardInterrupt:
    print("Process interrupted. Saving progress...")
    progress['index'] = idx  # Save the last processed index
    save_progress(checkpoint_file, progress)
    print(f"Progress saved. You can resume from iteration {idx + 1} by running the script again.")
else:
    # Save the final progress
    progress['index'] = idx + 1
    save_progress(checkpoint_file, progress)
    print("Processing completed. Final progress saved.")

import pickle

checkpoint_file = 'progress_checkpoint.pkl'  # Change this to your checkpoint filename

def load_progress(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    return None

progress = load_progress(checkpoint_file)

if progress is not None:
    print("Contents of the progress checkpoint:")
    print(progress['embeddings'][0])
else:
    print("No progress checkpoint found.")

embeddings = progress['embeddings']

embeddings[0]


import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


import bitsandbytes as bnb

from tqdm import tqdm
from operator import itemgetter

import json

from pytorch_metric_learning.losses import NTXentLoss

class config:
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 4

    model_name ='mrm8488/llama-2-coder-7b'
    text_embedding = 2048
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


        code1_embeddings = self.code1_encoder(torch.tensor(batch[0]))
        code2_embeddings = self.code1_encoder(torch.tensor(batch[1]))

        if train:
          code1_embeddings = self.code1_projection_head(code1_embeddings)
          code2_embeddings = self.code1_projection_head(code2_embeddings)

        return code1_embeddings, code2_embeddings

"""## Sanity Check"""

def count_common_elements(list1, list2):
    set1 = set([elem for sublist in list1 for elem in sublist])
    set2 = set([elem for sublist in list2 for elem in sublist])
    common_elements = set1 & set2
    return len(common_elements)

count_common_elements(train_data, eval_probs)

"""## Training (continued)"""

code1_encoder = MLPWithResidual(2048, 2048, 4)

contrastive_learner = SimCLR(code1_encoder=code1_encoder)

contrastive_learner_optimizer = optim.AdamW(contrastive_learner.parameters(), lr=1e-4)

torch.cuda.empty_cache()

def batch_generator(data, batch_size):
    current_index = 0
    data_length = len(data)

    while current_index < data_length:
        batch = data[current_index : current_index + batch_size]
        current_index += batch_size

        batch_list1 = [item[0] for item in batch]
        batch_list2 = [item[1] for item in batch]

        yield batch_list1, batch_list2

from sklearn.metrics import accuracy_score

def evaluate_accuracy(contrastive_learner, dataset_eval, threshold=0.5):
    # Set the model in evaluation mode
    contrastive_learner.eval()

    with torch.no_grad():
        all_predictions = []
        all_labels = []

        for idx, datapoint in enumerate(dataset_eval):
            code1_embeddings, code2_embeddings = contrastive_learner([datapoint[0],datapoint[1]], train=False)
            similarity = F.cosine_similarity(code1_embeddings, code2_embeddings, dim=-1)
            predictions = (similarity > threshold).float()
            all_predictions.extend(predictions.tolist())
            all_labels.extend([1]*2)
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)

    return accuracy, all_predictions, all_labels

epochs = 20
batch_size = 2
negatives_ratio = 15

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(contrastive_learner_optimizer, T_max=750, eta_min=1e-8)

loss_list = []
accs_list = []
preds_list = []
best_accuracy = 0
for epoch in range(epochs):
    progress_bar = tqdm(range(0, 1000000 , batch_size))

    dataset_val = batch_generator(embeddings[:50], 2)
    acc, preds, labels = evaluate_accuracy(contrastive_learner, dataset_val,.7)
    print(f'Validation Accuracy: {acc}')
    accs_list.append(acc)
    preds_list.append(preds)
    torch.cuda.empty_cache()
    if acc > best_accuracy and epoch > 5:
      torch.save(contrastive_learner.state_dict(), "contrastive_model_params.pth")
      best_accuracy=acc

    dataset_train = batch_generator(embeddings[50:], 16)
    for idx, datapoint in zip(progress_bar, dataset_train):
      contrastive_learner_optimizer.zero_grad()
      code1_embeddings, code2_embeddings = contrastive_learner([datapoint[0],datapoint[1]])
      loss = contrastive_learner.get_loss(code1_embeddings, code2_embeddings)
      loss_list.append(loss.item())


      loss.backward()
      contrastive_learner_optimizer.step()
      torch.cuda.empty_cache()
      progress_bar.set_description(f'Epoch: {epoch+1} | Running Loss: {sum(loss_list)/len(loss_list)} | Current Loss: {loss_list[-1]}')

      scheduler.step()
      torch.cuda.empty_cache()

state_dict = torch.load("contrastive_model_params.pth")
contrastive_learner.load_state_dict(state_dict)

"""#Evaluation"""

from sklearn.metrics import accuracy_score
import torch
import openai
import pickle
import os

def evaluate_accuracy(dataset_eval, threshold=0.8):
    saved_file = 'embeddings_labels.pkl'
    all_data = []

    # Load previous data if the file exists
    if os.path.exists(saved_file):
        with open(saved_file, 'rb') as file:
            all_data = pickle.load(file)
        print(f"Loaded {len(all_data)} previous data entries.")

    try:
        for idx, datapoint in enumerate(dataset_eval):
            if idx < len(all_data):
                continue  # Skip processed datapoints

            code1_embeddings = [
                openai.Embedding.create(
                    input=datapoint[0][0],
                    model="code-search-ada-code-001"
                )['data'][0]['embedding'],
                openai.Embedding.create(
                    input=datapoint[0][1],
                    model="code-search-ada-code-001"
                )['data'][0]['embedding'],
                openai.Embedding.create(
                    input=datapoint[0][2],
                    model="code-search-ada-code-001"
                )['data'][0]['embedding'],
                openai.Embedding.create(
                    input=datapoint[0][3],
                    model="code-search-ada-code-001"
                )['data'][0]['embedding'],
            ]

            code2_embeddings = [
                openai.Embedding.create(
                    input=datapoint[1][0],
                    model="code-search-ada-code-001"
                )['data'][0]['embedding'],
                openai.Embedding.create(
                    input=datapoint[1][1],
                    model="code-search-ada-code-001"
                )['data'][0]['embedding'],
                openai.Embedding.create(
                    input=datapoint[1][2],
                    model="code-search-ada-code-001"
                )['data'][0]['embedding'],
                openai.Embedding.create(
                    input=datapoint[1][3],
                    model="code-search-ada-code-001"
                )['data'][0]['embedding'],
            ]

            all_data.append({
                'code1_embeddings': code1_embeddings,
                'code2_embeddings': code2_embeddings,
                'labels': datapoint[2]
            })

            # Save embeddings and labels in a single file using pickle
            with open(saved_file, 'wb') as file:
                pickle.dump(all_data, file)

            print(f"Processed datapoint {idx + 1}.")

    except Exception as e:
        print(f"An error occurred: {e}. Process will resume from the last processed datapoint.")

    return all_data

data = evaluate_accuracy(contrastive_data_generator_eval(eval_probs, batch_size=1))

import pickle

def load_embeddings(saved_file='embeddings_labels.pkl'):
    try:
        with open(saved_file, 'rb') as file:
            all_data = pickle.load(file)
        return all_data
    except FileNotFoundError:
        print(f"File '{saved_file}' not found.")
        return []

# Load the embeddings and labels
embeddings_labels = load_embeddings()

# Print the loaded embeddings for the first datapoint
if embeddings_labels:
    first_datapoint = embeddings_labels[0]
    code1_embeddings = first_datapoint['code1_embeddings']
    code2_embeddings = first_datapoint['code2_embeddings']


    print("Code 1 Embeddings:")
    for embedding in code1_embeddings:
        print(embedding)
        break

    print("\nCode 2 Embeddings:")
    for embedding in code2_embeddings:
        print(embedding)
        break
else:
    print("No embeddings and labels loaded.")

import pickle

def load_embeddings(saved_file='embeddings_labels.pkl'):
    try:
        with open(saved_file, 'rb') as file:
            all_data = pickle.load(file)
        return all_data
    except FileNotFoundError:
        print(f"File '{saved_file}' not found.")
        return []

# Load the embeddings and labels
embeddings_labels = load_embeddings()

# Extract and convert data into separate lists
if embeddings_labels:
    code1_embeddings_list = []
    code2_embeddings_list = []
    labels_list = []

    for datapoint in embeddings_labels:
        code1_embeddings_list.append(datapoint['code1_embeddings'])
        code2_embeddings_list.append(datapoint['code2_embeddings'])
        labels_list.append(datapoint['labels'])


else:
    print("No embeddings and labels loaded.")

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F

contrastive_learner.eval()

all_similarities = []

with torch.no_grad():
    for code1, code2 in zip(code1_embeddings_list, code2_embeddings_list):
        code1_embeddings, code2_embeddings = contrastive_learner([code1, code2], train=False)
        similarity = F.cosine_similarity(code1_embeddings, code2_embeddings, dim=-1)
        all_similarities.extend(similarity.tolist())

all_similarities = np.array(all_similarities)


threshold = .8

predictions = (all_similarities > threshold).astype(int)

acc = accuracy_score(np.array(labels_list).flatten(), predictions)
precision = precision_score(np.array(labels_list).flatten(), predictions)
recall = recall_score(np.array(labels_list).flatten(), predictions)
f1 = f1_score(np.array(labels_list).flatten(), predictions)

print(acc, precision, recall, f1)