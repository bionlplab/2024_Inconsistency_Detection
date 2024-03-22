import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score

seeds = [10, 42, 1010, 1234, 2023]

def common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default='BioBERT', type=str, help="Pre-trained BERT model.")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Batch size train.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=30, type=int, help="Total number of training epochs", )
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    parser.add_argument("--seed", type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--output_dir", type=str, default='output.txt', help="Output file dir.")

    return parser

def calculate_metrics(pred, target):
    return {'precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'accuracy': accuracy_score(y_true=target, y_pred=pred)
            }