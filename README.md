# SCALE: Semantic Code Analysis via Learned Embeddings .

#### 3rd best paper on Artificial Intelligence track | presented at the 2023 International Conference on AI, Blockchain, Cloud Computing and Data Analytics

This repository holds the code and supplementary materials for SCALE: Semantic Code Analysis via Learned Embeddings. This research explores the efficacy of contrastive learning alongside large language models as a paradigm for developing a model capable of creating code embeddings indicative of code on a functional level.

## Abstract
Existing pre-trained models in NLP have demonstrated impressive success, surpassing previous benchmarks in various language-related tasks. However, when it comes to the field of code understanding, these models still face notable limitations. Code isomorphism, which deals with determining functional similarity between pieces of code, presents a challenging problem for NLP models. In this paper, we explore two approaches to code isomorphism. Our first approach, dubbed SCALE-FT, formulates the problem as a binary classification task, where we feed pairs of code snippets to a Large Language Model (LLM), using the embeddings to predict whether the given code segments are equivalent. The second approach, SCALE-CLR, adopts the SimCLR framework to generate embeddings for individual code snippets. By processing code samples with an LLM and observing the corresponding embeddings, we assess the similarity of two code snippets. These approaches enable us to leverage function-based code embeddings for various downstream tasks, such as code-optimization, code-comment alignment, and code classification. Our experiments on the CodeNet Python800 benchmark demonstrate promising results for both approaches. Notably, our SCALE-FT using Babbage-001 (GPT-3) achieves state-of-the-art performance, surpassing various benchmark models such as GPT-3.5 Turbo and GPT-4. Additionally, Salesforce's 350-million parameter CodeGen, when trained with the SCALE-FT framework, surpasses GPT-3.5 and GPT-4.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Prerequisites

Before running the code, make sure you have the following prerequisites:

- Python 3.x
- PyTorch
- Torchsummary
- Opencv
- Tqdm
- Imageio
- Numpy
- Peft
- Matplotlib 

## Installation

1. Clone this repository to your local machine.

```bash
git clone https://github.com/Jaso1024/Semantic-Code-Analysis.git
```

2. Dataset Coming soon

## Usage

For both frameworks (SCALE-FT and SCALE-CLR) all training files are located in the Training folder and evaluations in the Evaluation folder.

## Results

The results obtained from the experiments are as follows:

![image](https://github.com/Jaso1024/Semantic-Code-Analysis/assets/107654508/57f44eb8-8049-4695-9298-5ee712f63ff1)

The SCALE-FT framework with the Babbage version of GPT-3 as its base model achieved the best performance, with a 20% increase in f1-score when compared to GPT-4. Notably, even the 350 million parameter model, salesforce's codeegen, when trained with the SCALE-FT framework, outperforms GPT-4 in terms of accuracy, precision, and f1-score.
