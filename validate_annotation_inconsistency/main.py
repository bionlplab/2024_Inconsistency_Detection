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

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

MODELS = {'BERT': "bert-base-uncased", 
          'PubmedBERT': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
          'BioBERT': "dmis-lab/biobert-base-cased-v1.2"}

state_df = pd.read_csv('/data/state-list.csv')

def evaluate(args, test_dataloader, model, criterion, device, epoch, FILE_NAME):
    model.eval()
    total_loss = 0
    model_predictions, targets, pred_scores = [], [], []

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
    
    with open(FILE_NAME, 'a') as file:
        file.write('Evaluation loss: {}\n'.format(avg_loss))
        file.write('Precision: {:.5f}, Recall: {:.5f}, F1: {:.5f}\n'.format(result['macro/precision'], result['macro/recall'], result['macro/f1']))
        file.write('Accuracy: {:.5f} \n'.format(result['accuracy']))
    
    return avg_loss, result['macro/f1']

def train(args, train_dataloader, validation_dataloader, target_state_test_dataloader, other_states_test_dataloader, model, tokenizer, device, FILE_NAME):
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
        with open(FILE_NAME, 'a') as file:
            file.write('Epoch: {}, Training loss: {}.\n'.format(epoch, avg_loss))
            file.write('Validation: \n')

        val_loss, val_f1 = evaluate(args, validation_dataloader, model, criterion, device, epoch, FILE_NAME)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            
        with open(FILE_NAME, 'a') as file:
            file.write('----\n')
            file.write('Evaluation on target_state: \n')
        
        target_state_loss, target_state_f1 = evaluate(args, target_state_test_dataloader, model, criterion, device, epoch, FILE_NAME)
        
        with open(FILE_NAME, 'a') as file:
            file.write('----\n')
            file.write('Evaluation on other states: \n')
        
        other_states_loss, other_states_f1 = evaluate(args, other_states_test_dataloader, model, criterion, device, epoch, FILE_NAME)
        
        with open(FILE_NAME, 'a') as file:
            file.write('----------------\n')
    
    with open(FILE_NAME, 'a') as file:
        file.write('****************** BEST EPOCH: {} ****************** \n'.format(best_epoch))
            
if __name__ == '__main__':
    parser = common_args()
    args = parser.parse_args()
    device = torch.device('cuda', args.gpu_device)
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.bert_model])
    print('Device:', device)
    STATE_BLACKLIST = [88, 99]

    for CRISIS in ['FamilyRelationship', 'MentalHealth', 'PhysicalHealth']:
        STATE_ID_LIST, STATE_NAME_LIST = [], []
        DATASETS = {}
        for idx, row in tqdm(state_df.iterrows(), total=len(state_df)):
            state_id, state = row['STATEFIPS'], row['State']
            if state_id not in STATE_BLACKLIST:
                STATE_ID_LIST.append(state_id)
                STATE_NAME_LIST.append(state.replace(" ", ""))

                # read the data from target state, and other states
                DATASETS[state_id] = [f'/data/{CRISIS}/{state_id}.csv', f'/data/{CRISIS}/{state_id}_other_states.csv']

        for idx in range(len(STATE_ID_LIST)):
            state_id, state_name = STATE_ID_LIST[idx], STATE_NAME_LIST[idx]
            target_state = pd.read_csv(DATASETS[state_id][0], low_memory=False)

            # sample data from other states as in Figure 4, step 1 in the paper.
            other_states = pd.read_csv(DATASETS[state_id][1], low_memory=False).sample(n = int(4.8 * len(target_state))).reset_index(drop=True)
            other_states_main, other_states_extra = train_test_split(other_states, test_size=int(len(target_state)*0.8))

            print('*'*40)
            print(f'Dataset dataframe loaded. {state_id} set size {len(target_state)}, All others set size {len(other_states)}.')
            
            for seed in seeds:
                # create train, val, test sets as in Figure 4, step 1 in the paper.
                target_state_train, target_state_test = train_test_split(target_state, test_size=int(len(target_state)*0.2), random_state=seed)
                other_states_train, other_states_test = train_test_split(other_states_main, test_size=int(len(target_state)*4*0.2), random_state=seed)
                
                target_state_test, target_state_val = train_test_split(target_state_test, test_size=0.5, random_state=seed)
                other_states_test, other_states_val = train_test_split(other_states_test, test_size=0.5, random_state=seed)
                
                target_state_train_1, target_state_train_2 = train_test_split(target_state_train, test_size=0.5, random_state=seed)
                other_states_train_1, other_states_train_2 = other_states_train, other_states_extra
                        
                target_state_train_1_dataset, target_state_train_2_dataset, target_state_test_dataset, target_state_val_dataset = Dataset(target_state_train_1, CRISIS), Dataset(target_state_train_2, CRISIS), Dataset(target_state_test, CRISIS), Dataset(target_state_val, CRISIS)
                other_states_train_1_dataset, other_states_train_2_dataset, other_states_test_dataset, other_states_val_dataset = Dataset(other_states_train_1, CRISIS), Dataset(other_states_train_2, CRISIS), Dataset(other_states_test, CRISIS), Dataset(other_states_val, CRISIS)

                print('*'*40)
                print('Dataset split complete. target_state train 1 size {}, target_state train 2 size {}, target_state test size {}.'.format(len(target_state_train_1), len(target_state_train_2), len(target_state_test)))
                print('Dataset split complete. Others train 1 size {}, Others train 2 size {}, Others test size {}.'.format(len(other_states_train_1), len(other_states_train_2), len(other_states_test)))
                
                validation_dataset = ConcatDataset([target_state_val_dataset, other_states_val_dataset])
                validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.train_batch_size, pin_memory=True)
                target_state_test_dataloader = torch.utils.data.DataLoader(target_state_test_dataset, batch_size=args.train_batch_size, pin_memory=True)
                other_states_test_dataloader = torch.utils.data.DataLoader(other_states_test_dataset, batch_size=args.train_batch_size, pin_memory=True)
                
                print('*'*40)
                print('Dataset loader loaded.')

                # Create three combinations of training data: PureOthers, Others+Target, Target+Others, as shown in Figure 4, step 1 in the paper.
                print('*'*40)
                # Target + Others
                print(f'{state_name}_Train, Other_Train_1')
                FILE_NAME = '{}_{}_{}_T_Other_T1_{}.txt'.format(CRISIS, state_id, state_name, seed)
                model = BertClassifier(bert=MODELS[args.bert_model])
                model.to(device)
                with open(FILE_NAME, 'w') as file:
                    file.write('**'*20 + '\n')
                    file.write(f'Training with: {state_name}_Train, Other_Train_1 \n')
                    file.write('**'*20 + '\n')
                target_state_train_other_states_train_1 = ConcatDataset([target_state_train_1_dataset, target_state_train_2_dataset, other_states_train_1_dataset])
                train_dataloader = torch.utils.data.DataLoader(target_state_train_other_states_train_1, batch_size=args.train_batch_size, pin_memory=True)
                train(args, train_dataloader, validation_dataloader, target_state_test_dataloader, other_states_test_dataloader, model, tokenizer, device, FILE_NAME)
                
                print('*'*40)
                # PureOthers
                print('Other_Train_1, Other_Train_2')
                model = BertClassifier(bert=MODELS[args.bert_model])
                model.to(device)
                FILE_NAME = '{}_{}_Other_T_{}.txt'.format(CRISIS, state_id, seed)
                with open(FILE_NAME, 'w') as file:
                    file.write('**'*20 + '\n')
                    file.write('Training with: Other_Train_1, Other_Train_2 \n')
                    file.write('**'*20 + '\n')
                other_states_train_1_other_states_train_2 = ConcatDataset([other_states_train_1_dataset, other_states_train_2_dataset])
                train_dataloader = torch.utils.data.DataLoader(other_states_train_1_other_states_train_2, batch_size=args.train_batch_size, pin_memory=True)
                train(args, train_dataloader, validation_dataloader, target_state_test_dataloader, other_states_test_dataloader, model, tokenizer, device, FILE_NAME)
                
                print('*'*40)
                # Others + Target
                print(f'Other_Train_1, {state_name}_Train')
                FILE_NAME = '{}_{}_Other_T1_{}_T_{}.txt'.format(CRISIS, state_id, state_name, seed)
                model = BertClassifier(bert=MODELS[args.bert_model])
                model.to(device)
                with open(FILE_NAME, 'w') as file:
                    file.write('**'*20 + '\n')
                    file.write(f'Training with: Other_Train_1, {state_name}_Train \n')
                    file.write('**'*20 + '\n')
                other_states_train_1_target_state_train = ConcatDataset([other_states_train_1_dataset, target_state_train_1_dataset, target_state_train_2_dataset])
                train_dataloader = torch.utils.data.DataLoader(other_states_train_1_target_state_train, batch_size=args.train_batch_size, pin_memory=True)
                train(args, train_dataloader, validation_dataloader, target_state_test_dataloader, other_states_test_dataloader, model, tokenizer, device, FILE_NAME)
    