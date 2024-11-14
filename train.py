### This is run when you want to select the parameters from the parameters file
import transformers 
import torch
from knockknock import slack_sender
# from transformers import *
import glob 
from transformers import BertTokenizer,RobertaTokenizerFast
from transformers import BertForSequenceClassification, AdamW, BertConfig
import random 
import pandas as pd
from transformers import BertTokenizer
from Models.utils import masked_cross_entropy,fix_the_random,format_time,save_normal_model,save_bert_model,return_params
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
from TensorDataset.dataLoader import combine_features
from Preprocess.dataCollect import set_name, collect_data_trace
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import matplotlib.pyplot as plt
import time
import os
from transformers import BertTokenizer
import GPUtil
from sklearn.utils import class_weight
import json
from Models.bertModels_pooling import *
from Models.otherModels import *
import sys
import time
# from waiting import wait
from sklearn.preprocessing import LabelEncoder
import numpy as np
import threading
import argparse
import ast
from rank_metrics import compute_metrics, plot_metrics
from trace_params import set_params
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, BertTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

### gpu selection algo
def get_gpu():
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 0.1, maxMemory = 0.07, includeNan=False, excludeID=[], excludeUUID=[])
        if len(tempID) > 0:
            print("Found a gpu")
            print('We will use the GPU:',tempID[0],torch.cuda.get_device_name(tempID[0]))
            deviceID=tempID
            return deviceID
        else:
            time.sleep(5)
#    return flag,deviceID


##### selects the type of model
def select_model(params,embeddings, tokenizer):
    if(params['bert_tokens']):
        if(params['what_bert']=='weighted'):
            model = SC_weighted_BERT.from_pretrained(
            params['model_name'], # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = params['num_classes'], # The number of output labels
            output_attentions = True, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            hidden_dropout_prob=params['dropout_bert'],
            params=params,
            )
        else:
            print("Error in bert model name!!!!")
        return model
    else:
        text=params['model_name']
        if(text=="birnn"):
            model=BiRNN(params,embeddings)
        elif(text == "birnnatt"):
            model=BiAtt_RNN(params,embeddings,return_att=False,)
        elif(text == "birnnscrat"):
            model=BiAtt_RNN(params,embeddings,return_att=True)
        elif(text == "cnn_gru"):
            model=CNN_GRU(params,embeddings)
        elif(text == "lstm_bad"):
            model=LSTM_bad(params)
        else:
            print("Error in model name!!!!")
        return model


def Eval_phase(params, which_files='test', model=None, test_dataloader=None, device=None):
    if(params['is_model']==True):
        print("model previously passed")
        model.eval()
    else:
        return 1


    print("Running eval on ",which_files,"...")
    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables 
    
    true_labels = []
    pred_labels = []
    pred_proba_all = []
    post_ids = []
    # Evaluate data for one epoch
    with torch.no_grad():  # fix extra memory usage when evaluating
        for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)


            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention vals
            #   [2]: attention mask
            #   [3]: labels 
            b_text_ids = torch.stack(batch[2]).to(device)
            b_text_att = torch.stack(batch[3]).to(device)
            b_text_mask = torch.stack(batch[4]).to(device)
            b_code_num = [len(batch[5][x]) for x in range(len(batch[5]))]
            #b_code_ids = torch.cat(batch[5]).to(device)
            #b_code_att = torch.cat(batch[6]).to(device)
            #b_code_mask = torch.cat(batch[7]).to(device)
            b_code_ids_padded = pad_sequence(batch[5], batch_first=True, padding_value=0).data.to(device)
            b_code_ids_packed = pack_padded_sequence(b_code_ids_padded, lengths=b_code_num, batch_first=True, enforce_sorted=False)
            b_code_att_padded = pad_sequence(batch[6], batch_first=True, padding_value=0).data.to(device)
            b_code_att_packed = pack_padded_sequence(b_code_att_padded, lengths=b_code_num, batch_first=True, enforce_sorted=False)
            b_code_mask_padded = pad_sequence(batch[7], batch_first=True, padding_value=0).data.to(device)
            b_code_mask_packed = pack_padded_sequence(b_code_mask_padded, lengths=b_code_num, batch_first=True, enforce_sorted=False)

            b_post_ids = batch[1]
            b_labels = torch.stack(batch[0]).to(device)

            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        
            outputs = model(input_ids=b_text_ids, 
                text_att=b_text_att,
                text_mask=b_text_mask, 
                code_ids=b_code_ids_packed,
                code_att=b_code_att_packed,
                code_mask=b_code_mask_packed,
                code_lengths=b_code_num,
                labels=None,
                device=device)
            
            pred_proba = F.softmax(outputs[0], dim=1)
            # Move pred_proba and labels to CPU
            pred_proba = pred_proba.detach().cpu().numpy()  # shape [batch_size, num_labels]
            label_ids = b_labels.to('cpu').numpy()
            
            if step == 0:
                print(f'shape of pred_proba: {pred_proba.shape}')
                print(f'shape of label_ids: {label_ids.shape}')
                
            # Calculate the accuracy for this batch of test sentences.
            # Accumulate the total accuracy.
            pred_labels += list(np.argmax(pred_proba, axis=1).flatten())
            true_labels += list(label_ids.flatten())
            pred_proba_all += list(pred_proba) # list of [num_classes] arrays
            post_ids += b_post_ids
            
    
    if params['num_classes'] > 1:
        pred_proba_all_final = []
        for pred_proba in pred_proba_all:
            pred_proba_all_final.append(pred_proba)
    else:
        pred_proba_all_final = [pred_proba for pred_proba in pred_proba_all] # convert pred_proba to probabilities
        
    if params['num_classes'] == 2:
        probs_all_final = [pred_proba[1] for pred_proba in pred_proba_all_final]
    elif params['num_classes'] == 1:
        probs_all_final = [pred_proba[0] for pred_proba in pred_proba_all_final]
    metric_dict = compute_metrics(scores=probs_all_final, labels=true_labels, topk=1, cves=post_ids)
    testf1 = metric_dict['f1_best']
    testacc = -9999999
    testrocauc = metric_dict['auc']
    testprecision = metric_dict['precision@k'][0]
    testrecall = metric_dict['recall@k'][0]
    testmap = metric_dict['MAP']
        
    if(params['is_model'] == True):
        # Report the final accuracy for this validation run.
        print(" Accuracy: {0:.2f}".format(testacc))
        print(" Fscore: {0:.2f}".format(testf1))
        print(" Precision: {0:.2f}".format(testprecision))
        print(" Recall: {0:.2f}".format(testrecall))
        print(" Roc Auc: {0:.2f}".format(testrocauc))
        print(" MAP: {0:.2f}".format(testmap))
        print(" Test took: {:}".format(format_time(time.time() - t0)))
        #print(ConfusionMatrix(true_labels,pred_labels))

    return testf1, testacc, testprecision, testrecall, testrocauc, testmap, pred_proba_all_final




def train_model(params,device):

    print('Loading BERT tokenizer...')
    if "codebert" in params['model_name'] or "unixcoder" in params['model_name']:
        tokenizer = RobertaTokenizerFast.from_pretrained(params['model_name'], do_lower_case=False)
        #tokenizer.add_tokens(["[NEWLINE]"])
    elif "bert" in params["model_name"]:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)

    embeddings=None
    #raise Exception((params['bert_tokens']))
    if params['problem'] == 'trace':
        print("Loading data from trace")
        train = collect_data_trace(params, "train")
        val = collect_data_trace(params, "valid")
        test = collect_data_trace(params, "test") 
        print("train:", len(train))
        print("test:", len(test))
        print("valid:", len(val))
    else:
        raise ValueError("Problem not found, only trace and hatexplain are supported")
    
    print("Training set size: ",len(train))
    print("Validation set size: ",len(val))
    print("Test set size: ",len(test))
    # save_bert_model(None, None, params) # save the model name in the file
    try:
        print(train.head(5))
        print(f'trian set label distribution: {train["Label"].sum()/len(train)}')
        print(f'val set label distribution: {val["Label"].sum()/len(val)}')
        print(f'test set label distribution: {test["Label"].sum()/len(test)}')
    except AttributeError:
        print(train[:5])
    except TypeError:
        pass
    
    # raise Exception('stop')
    
    if params['problem'] == 'trace':
        train_post_ids = list(train['Post_id'])
        train = train.iloc[:, 1:]
        train = list(train.itertuples(index=False, name=None))
        
        val_post_ids = list(val['Post_id'])
        val = val.iloc[:, 1:]
        val = list(val.itertuples(index=False, name=None))
        
        test_post_ids = list(test['Post_id'])
        test = test.iloc[:, 1:]
        test = list(test.itertuples(index=False, name=None))
    else:
        train_post_ids, val_post_ids, test_post_ids = None, None, None
        
    if params['auto_weights']:
        y_test = [ele[0] for ele in test] 
        #print(f'weights: {y_test[:100]}')
#         print(y_test)
        encoder = LabelEncoder()
        encoder.classes_ = np.load(params['class_names'], allow_pickle=True)
        print(encoder.classes_)
        params['weights'] = class_weight.compute_class_weight('balanced', np.unique(y_test), y_test).astype('float32') 
        #print(f'weights: {params["weights"]}')
        # raise Exception('stop, check auto weights')
    
    train_dataloader = combine_features(train, params, post_ids=train_post_ids, is_train=True)   
    validation_dataloader = combine_features(val, params, post_ids=val_post_ids, is_train=False, batch_size=params["batch_size_test"])
    test_dataloader = combine_features(test, params, post_ids=test_post_ids, is_train=False, batch_size=params["batch_size_test"])
    
    # raise Exception('stop for validation')
    model = select_model(params, embeddings, tokenizer)
    
    if (params["device"]=='cuda'):
        print(f'model moved to {params["device"]}')
        model.cuda()
    optimizer = AdamW(model.parameters(),
                  lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
                )

    # Number of training epochs (authors recommend between 2 and 4)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * params['epochs']

    # Create the learning rate scheduler.
    if (params['bert_tokens']):
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps/10),                     num_training_steps = total_steps)

    # Set the seed value all over the place to make this reproducible.
    fix_the_random(seed_val = params['random_seed'])
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    if (params['bert_tokens']):
        bert_model = params['model_name']
        name_one=bert_model
        
    best_val_fscore=0
    best_test_fscore=0

    best_val_roc_auc=0
    best_test_roc_auc=0
    
    best_val_precision=0
    best_test_precision=0
    
    best_val_recall=0
    best_test_recall=0

    best_val_map=0
    best_test_map=0
    
    # For each epoch...
    precision_at_1_list_train = []
    precision_at_1_list_valid = []
    precision_at_1_list_test = []
    recall_at_1_list_train = []
    recall_at_1_list_valid = []
    recall_at_1_list_test = []
    map_list_train = []
    map_list_valid = []
    map_list_test = []
    
    print(f'Training... on dataset: {params["data_file"]}')
    # pdb.set_trace()

    for epoch_i in range(0, int(params['epochs'])):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()
        #save_bert_model(model, tokenizer, params)
        
        c = 0
        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # val_fscore, val_accuracy, val_precision, val_recall, val_roc_auc, val_map, _= Eval_phase(params,'val',model, validation_dataloader,device)

                # print(val_fscore, val_precision, val_recall, val_roc_auc)
                ## Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
            # c += 1
            # if c > 100:  # for debugging
            #     break
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention vals
            #   [2]: attention mask
            #   [3]: labels 
            b_text_ids = torch.stack(batch[2]).to(device)
            b_text_att = torch.stack(batch[3]).to(device)
            b_text_mask = torch.stack(batch[4]).to(device)

            b_code_num = [len(batch[5][x]) for x in range(len(batch[5]))]
            #b_code_ids = torch.cat(batch[5]).to(device)
            #b_code_att = torch.cat(batch[6]).to(device)
            #b_code_mask = torch.cat(batch[7]).to(device)
            b_code_ids_padded = pad_sequence(batch[5], batch_first=True, padding_value=0).data.to(device)
            b_code_ids_packed = pack_padded_sequence(b_code_ids_padded, lengths=b_code_num, batch_first=True, enforce_sorted=False)
            b_code_att_padded = pad_sequence(batch[6], batch_first=True, padding_value=0).data.to(device)
            b_code_att_packed = pack_padded_sequence(b_code_att_padded, lengths=b_code_num, batch_first=True, enforce_sorted=False)
            b_code_mask_padded = pad_sequence(batch[7], batch_first=True, padding_value=0).data.to(device)
            b_code_mask_packed = pack_padded_sequence(b_code_mask_padded, lengths=b_code_num, batch_first=True, enforce_sorted=False)

            b_labels = torch.stack(batch[0]).to(device)

            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()     
            outputs = model(input_ids=b_text_ids, 
                text_att=b_text_att,
                text_mask=b_text_mask, 
                code_ids=b_code_ids_packed,
                code_att=b_code_att_packed,
                code_mask=b_code_mask_packed,
                code_lengths=b_code_num,
                labels=b_labels,
                device=device)

            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            # print(loss)
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            if(params['bert_tokens']):
                scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        print('avg_train_loss', avg_train_loss)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        val_fscore, val_accuracy, val_precision, val_recall, val_roc_auc, val_map, _= Eval_phase(params,'val',model, validation_dataloader,device)
        test_fscore, test_accuracy, test_precision, test_recall, test_roc_auc, test_map, pred_proba_all_final = Eval_phase(params,'test',model,test_dataloader, device)
        
        precision_at_1_list_valid += [val_precision]
        precision_at_1_list_test += [test_precision]

        recall_at_1_list_valid += [val_recall]
        recall_at_1_list_test += [test_recall]

        map_list_valid += [val_map]
        map_list_test += [test_map]
    
        if(val_map > best_val_map):
            print(val_fscore,best_val_fscore)
            best_val_fscore=val_fscore
            best_test_fscore=test_fscore
            best_val_roc_auc = val_roc_auc
            best_test_roc_auc = test_roc_auc
            
            best_val_precision = val_precision
            best_test_precision = test_precision
            best_val_recall = val_recall
            best_test_recall = test_recall

            best_val_map = val_map
            best_test_map = test_map
            
            if(params['bert_tokens']):
                save_bert_model(model, tokenizer, params)
            else:
                print("Saving model")
                save_normal_model(model, params)

    # Plot the learning curve.
    plot_metrics(precision_at_1_list_valid, precision_at_1_list_test, title=f'Precision@1_lambda={params["att_lambda"]}', metric_name='Precision@1')
    
    plot_metrics(recall_at_1_list_valid, recall_at_1_list_test, title=f'Recall@1_lambda={params["att_lambda"]}', metric_name='Recall@1')
    plot_metrics(map_list_valid, map_list_test, title=f'MAP_lambda={params["att_lambda"]}', metric_name='MAP')
    
    print('best_val_fscore',best_val_fscore)
    print('best_test_fscore',best_test_fscore)
    print('best_val_rocauc',best_val_roc_auc)
    print('best_test_rocauc',best_test_roc_auc)
    print('best_val_precision',best_val_precision)
    print('best_test_precision',best_test_precision)
    print('best_val_recall',best_val_recall)
    print('best_test_recall',best_test_recall)
    print('best_val_map',best_val_map)
    print('best_test_map',best_test_map)
        
    del model
    torch.cuda.empty_cache()
    return 1









params_data={
    'include_special':False, 
    'bert_tokens':False,
    'type_attention':'softmax',
    'set_decay':0.1,
    'majority':2,
    'max_length':128,
    'variance':10,
    'window':4,
    'alpha':0.5,
    'p_value':0.8,
    'method':'additive',
    'decay':False,
    'normalized':False,
    'not_recollect':True,
}

#"birnn","birnnatt","birnnscrat","cnn_gru"


common_hp={
    'is_model':True,
    'logging':'local',  ###neptune /local
    'learning_rate':0.1,  ### learning rate 2e-5 for bert 0.001 for gru
    'epsilon':1e-8,
    'batch_size':16,
    'to_save':True,
    'epochs':10,
    'auto_weights':True,
    'weights':[1.0,1.0,1.0],
    'model_name':'birnnscrat',
    'random_seed':42,
    'num_classes':3,
    'att_lambda':100,
    'device':'cuda',
    'train_att':True

}


params_other = {
        "vocab_size": 0,
        "padding_idx": 0,
        "hidden_size":64,
        "embed_size":0,
        "embeddings":None,
        "drop_fc":0.2,
        "drop_embed":0.2,
        "drop_hidden":0.1,
        "train_embed":False,
        "seq_model":"gru",
        "attention":"softmax"
}


if(params_data['bert_tokens']):
    for key in params_other:
        params_other[key]='N/A'


def Merge(dict1, dict2, dict4): 
    res = {**dict1, **dict2, **dict4} 
    return res 

params = Merge(params_data,common_hp,params_other)


dict_data_folder={
      '2':{'data_file':'Data/dataset.json','class_label':'Data/classes_two.npy'},
      '3':{'data_file':'Data/dataset.json','class_label':'Data/classes.npy'}
}

if __name__=='__main__': 
    args = set_params().parse_args()
    params = return_params(args.path, args, load_trained_model=False)     
    
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
        
    train_model(params, device)

















