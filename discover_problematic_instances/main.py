import random
import torch
import os
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.optim import Adam
from torch.utils.data import ConcatDataset
from data import Dataset
from model import BertClassifier
from utils import common_args, calculate_metrics, seeds
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
MODELS = {'BERT': "bert-base-uncased", 
          'PubmedBERT': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
          'BioBERT': "dmis-lab/biobert-base-cased-v1.2"}

CRISIS = 'MentalHealth'
target_state_id = # the state id of the target state

def evaluate(args, test_dataloader, model, criterion, device, epoch):
    model.eval()
    total_loss = 0
    model_predictions, targets = [], []
    pred_scores = []

    for step, batch in enumerate(test_dataloader):
        report_input_ids, report_masks, labels = batch[0]['input_ids'].to(device), batch[0]['attention_mask'].to(device), batch[1].to(device)
            
        inputs = {'input_ids': report_input_ids.squeeze(1),
                  'attention_mask': report_masks.squeeze(1)}
        
        preds = model(**inputs)
        loss = criterion(preds, labels.float().unsqueeze(1))
        total_loss = total_loss + loss.item()
        
        pred_scores.extend(preds.detach().cpu().numpy())
        preds = torch.sigmoid(preds)
        preds = torch.round(preds)
        model_predictions.extend(preds.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())

    result = calculate_metrics(np.array(model_predictions), np.array(targets))
    avg_loss = total_loss / len(test_dataloader)
    
    with open(args.output_dir, 'a') as file:
        file.write('Evaluation loss: {}\n'.format(avg_loss))
        file.write('Precision: {:.5f}, Recall: {:.5f}, F1: {:.5f}\n'.format(result['macro/precision'], result['macro/recall'], result['macro/f1']))
        file.write('Accuracy: {:.5f} \n'.format(result['accuracy']))
    
    return avg_loss, result['macro/f1']

def count_wrong_predictions(args, test_dataloader, model, criterion, device, epoch):
    model.eval()
    total_loss = 0
    pred_scores = []
    wrong_ids = []

    for step, batch in enumerate(test_dataloader):
        report_input_ids, report_masks, labels, ids = batch[0]['input_ids'].to(device), batch[0]['attention_mask'].to(device), batch[1].to(device), batch[2].to(device)
            
        inputs = {'input_ids': report_input_ids.squeeze(1),
                  'attention_mask': report_masks.squeeze(1)}
        
        preds = model(**inputs)
        pred_scores.extend(preds.detach().cpu().numpy())
        preds = torch.sigmoid(preds)
        preds = torch.round(preds)
        
        for i in range(len(preds)):
            if preds[i] != labels[i]:
                wrong_ids.append(ids[i])
    
    return wrong_ids

def train(args, train_dataloader, test_dataloader, model, tokenizer, device, fold_count, seed):
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss().to(device)
    best_f1, best_epoch = 0, -1
    
    for epoch in tqdm(range(args.num_train_epochs)):
        print('Training, Epoch: {}'.format(epoch))
        total_loss = 0
        model.train()
        
        for step, batch in tqdm(enumerate(train_dataloader)):
            report_input_ids, report_masks, labels = batch[0]['input_ids'].to(device), batch[0]['attention_mask'].to(device), batch[1].to(device)
            
            inputs = {'input_ids': report_input_ids.squeeze(1),
                      'attention_mask': report_masks.squeeze(1)}
            
            preds = model(**inputs)
            loss = criterion(preds, labels.float().unsqueeze(1))
            total_loss = total_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_dataloader)
        with open(args.output_dir, 'a') as file:
            file.write('Epoch: {}, Training loss: {}.\n'.format(epoch, avg_loss))
            file.write('Evaluation: \n')

        loss, f1 = evaluate(args, test_dataloader, model, criterion, device, epoch)
        
        if f1>best_f1:
            best_f1 = f1
            best_epoch = epoch
            wrong_ids = count_wrong_predictions(args, test_dataloader, model, criterion, device, epoch)
            
            with open('{}_{}_wrong_ids.txt'.format(seed, fold_count), 'w') as write_file:
                for i in wrong_ids:
                    write_file.write('{}\n'.format(i))
        
    with open('{}_{}_wrong_ids.txt'.format(seed, fold_count), 'w') as write_file:
        for i in wrong_ids:
            write_file.write('{}\n'.format(i))
            
if __name__ == '__main__':
    parser = common_args()
    args = parser.parse_args()
    device = torch.device('cuda', args.gpu_device)
    print('Device:', device)
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.bert_model])

    target_state = pd.read_csv(f'/data/{CRISIS}/{target_state_id}.csv')
    other_states = pd.read_csv(f'/data/{CRISIS}/{target_state_id}_other_states.csv').sample(n=4*len(target_state)).reset_index(drop=True) 

    df = pd.concat([target_state, other_states], ignore_index=True)
    print('*'*40)
    print('Dataset dataframe loaded. Dataset size {}.'.format(len(df)))
    
    for seed in seeds:
        kf = KFold(n_splits = 5, shuffle = True, random_state = seed)
        fold_count = 0
        for result in kf.split(df):
            train_df = df.iloc[result[0]]
            test_df =  df.iloc[result[1]]
            train_dataset, test_dataset = Dataset(train_df, CRISIS), Dataset(test_df, CRISIS)

            print('*'*40)
            print('Dataset split complete. Train size {}, Test size {}.'.format(len(train_dataset), len(test_dataset)))

            train_dataloader, test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True), torch.utils.data.DataLoader(test_dataset, batch_size=args.train_batch_size, pin_memory=True)
            print('*'*40)
            print('Dataset loader loaded.')
           
            model = BertClassifier(bert=MODELS[args.bert_model], n_classes=1)
            model.to(device)
            train(args, train_dataloader, test_dataloader, model, tokenizer, device, fold_count, seed)    
            fold_count += 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    