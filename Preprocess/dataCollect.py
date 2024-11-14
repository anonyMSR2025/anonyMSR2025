import pandas as pd
from glob import glob
import json
from tqdm import tqdm_notebook,tqdm
from difflib import SequenceMatcher
from collections import Counter
import sys
import os
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
from Preprocess.process_ranktoken_attentions import tokenize_with_gt_attention_mask
#from Preprocess.preProcess import ek_extra_preprocess
from Preprocess.attentionCal import aggregate_attention
from transformers import BertTokenizer
from Preprocess.utils import CheckForGreater,most_frequent
from Preprocess.preProcess import *   
from transformers import BertTokenizer
from os import path
import pickle
import numpy as np
import re

from transformers import BertTokenizerFast, RobertaTokenizerFast, RobertaTokenizer

def set_name(params, is_trained=False):
    file_name = 'Trace'
    if is_trained:
        file_name += '_' + params["path_files"].replace("/", "-")
    else:
        file_name += '_' + params["model_name"].replace("/", "-")
    return file_name


def get_annotated_data_trace(params: dict, data_file: str, fold:str):
    '''
    read one data from 
    '''
    #temp_read = pd.read_pickle(params['data_file'])
    # tokenizer = None
    # train_data, valid_data, test_data = [], [], []
    import random
    random.seed(42)
    rt_dataset = []
    print('data_file', data_file)
    c = 0
    # 在训练时发现如果每个example都用text+code，test的时候只用code，会导致proba接近于0，这说明不对，因为人是可以只通过code来识别的，但训练的模型找到了只看text的捷径。如果用augmentation，在训练时加入一些只有code的example，会不会提高模型对code的理解？
    if params["is_augment"] == "code" and fold == "train":
        augment_options = ["origin", "code"]
    elif params["is_augment"] == "both" and fold == "train":
        augment_options = ["origin", "code", "text"]
    else:
        augment_options = ["origin"]

    with open(data_file, 'r') as fp:
        total_data = json.load(fp)
        for cve_and_commit, one_data in tqdm(total_data.items(), desc='Reading json data', total=len(total_data)):
            # print(one_data.keys()) 
            
            for is_augment in augment_options:
                temp={}
                temp['Post_id'] = cve_and_commit
                temp['Label'] = one_data['label']
                temp["sorted_idx"] = one_data["sorted_idx"]

                num_of_diff_code = len(one_data["diff_ids"])

                for suffix in ["", "_code"]:
                    if suffix == "":
                        temp['Text' + suffix] = [item for item in one_data['cve_desc_ids']]
                        temp['Attention' + suffix] = [0 for item in one_data['cve_desc_attention']]
                        temp['attention_pred_mask' + suffix] = [0 for item in one_data['cve_desc_attention']]
                    else:
                        temp['Text' + suffix] = []
                        temp["Attention" + suffix] = []
                        temp["attention_pred_mask" + suffix] = []
                        for x in range(num_of_diff_code):
                            temp["Text" + suffix].append([item for item in one_data['cve_desc_ids']])
                            temp['Attention' + suffix].append([0 for item in one_data['cve_desc_attention']])
                            temp['attention_pred_mask' + suffix].append([0 for item in one_data['cve_desc_attention']])
                
                    this_ids = one_data["commit_msg_ids"] if suffix == "" else one_data["diff_ids"]
                    this_attention = one_data["commit_msg_attention"] if suffix == "" else one_data["diff_attention"]

                    if suffix == "":
                        if is_augment in ["origin", "text"]:
                            temp['Text' + suffix] += [item for item in this_ids]
                            temp['Attention' + suffix] += this_attention # [1 if item > 0 else 0 for item in this_attention]
                            temp['attention_pred_mask' + suffix] += [1 for item in this_attention]
                        elif is_augment == "code":
                            temp['Text' + suffix] += [0, 2]
                            temp['Attention' + suffix] += [0, 0]
                            temp['attention_pred_mask' + suffix] += [1, 1]
                    else:
                        for x in range(num_of_diff_code):
                            if is_augment in ["origin", "code"]:
                                temp['Text' + suffix][x] += [item for item in this_ids[x]]
                                temp['Attention' + suffix][x] += this_attention[x] # [1 if item > 0 else 0 for item in this_attention[x]]
                                temp['attention_pred_mask' + suffix][x] += [1 for item in this_attention[x]]
                            elif is_augment == "text":
                                temp['Text' + suffix][x] += [0, 2]
                                temp['Attention' + suffix][x] += [0, 0]
                                temp['attention_pred_mask' + suffix][x] += [1, 1]
                
                if is_augment == "origin":
                    rt_dataset.append(temp)  
                elif is_augment in ["code", "text"]:
                    if one_data['label'] == 1:  # 如果是pos example，只有code也应该能识别
                        rt_dataset.append(temp)
                    else:
                        next_ran_float = random.uniform(0, 1)
                        if next_ran_float < 0.33:
                            rt_dataset.append(temp) # 需要加neg，全部加太多，只加1.2万个
            
    rt_dataset = pd.DataFrame(rt_dataset)
    # print(f'len(rt_dataset): {len(rt_dataset)}')
    return rt_dataset

def returnMask_trace(row, params, tokenizer, id_to_token_map, suffix):
    ## tokenize the text
    #rationale_colname = "rationales" + suffix
    #text_colname = "text" + suffix
    #rationales = row[rationale_colname][0]
    #assert len(rationales) == len(row[text_colname])
    ## remove changeline and <s> for each token in original_text_list:
    #new_text_list = []
    #new_rationale_list = []
    #is_last_end_with_space = False
    #for token_idx in range(len(rationales)):
    #    each_token = row[text_colname][token_idx]
    #    each_rationale = rationales[token_idx]
    #    #if suffix == "":
    #    #    new_token = re.sub("\s+", " ", each_token.replace("\n", ". ").replace(". </s>", ".</s>")) # to match before/after char len, need to remove extra whitespace
    #    #else:
    #    #    new_token = re.sub("\s+", " ", each_token.replace("\n", "[NEWLINE]"))
    #    new_token = re.sub(r'[^\x00-\x7F]+', '', new_token) # to match before/after character len, need to remove non-ascii code 
    #    #if row[text_colname][token_idx] == "</s>":
    #    #    new_text_list[-1] = new_text_list[-1].rstrip()
    #    #if token_idx + 1 < len(row["text"]) and row["text"][token_idx + 1] == "</s>": # remove 2 whitespaces
    #    #    new_token = new_token.rstrip()
    #    #if is_last_end_with_space: 
    #    #    if new_token == " ": continue
    #    #    new_token = new_token.lstrip()
    #    if len(new_token) > 0:
    #        new_text_list.append(new_token)
    #        new_rationale_list.append(each_rationale)
    #    #    is_last_end_with_space = new_token[-1] == " "
    #original_text = "".join(new_text_list).encode("utf-8").decode("utf-8")
    ## raise Exception(f'stop to check data: {original_text}')
    #tokens = custom_tokenize(original_text, tokenizer)
    #original_text_len = len(original_text)

    #start_idx = end_idx = 0
    #start_end_idx_list = [0] * original_text_len
    #for idx in range(len(new_text_list)):
    #    this_char_len = len(new_text_list[idx])
    #    end_idx = start_idx + this_char_len
    #    if new_rationale_list[idx] == 1:
    #        for char_idx in range(start_idx, end_idx):
    #            start_end_idx_list[char_idx] = 1
    #    start_idx += this_char_len

    ## get the attention mask
    #tokens_str_len = [len(id_to_token_map[x]) for x in tokens] 

    #attention_mask = []
    #assert sum(tokens_str_len) == original_text_len
    #start_idx = end_idx = 0
    #for idx in range(len(tokens_str_len)):
    #    this_char_len = tokens_str_len[idx]
    #    end_idx = start_idx + this_char_len
    #    if sum(start_end_idx_list[start_idx:end_idx]) > 0:
    #        attention_mask.append(1)
    #    else:
    #        attention_mask.append(0)
    #    start_idx += this_char_len

    # get the segment mask
    segment_mask = []
    flag = 0
    for idx in range(len(tokens)):
        if tokens[idx] == 2:
            flag = 1 - flag
        segment_mask.append(flag)
    return tokens, attention_mask, segment_mask


# data: pandas dataframe
def get_training_data_trace(data, params, tokenizer, is_train=False):
    '''input: data is a dataframe text ids attentions labels column only'''
    '''output: training data in the columns Post_id, text, attention and labels '''

    text_list=[]
    attention_list=[]
    attention_pred_mask_list=[]
    
    text_code_list = []
    attention_code_list = []
    attention_code_pred_mask_list=[]

    count=0
    count_confused=0
    print('total_data', len(data))

    # shuffle data to exam more data manually
    # data = data.sample(frac=1).reset_index(drop=True)
    c = 0
    for index, row in tqdm(data.iterrows(), total=len(data)):
        for suffix in ["_code", ""]:
            tokens_all, attention_masks, attention_pred_mask = returnMask_trace(row, params, tokenizer, id_to_token_map, suffix)
            assert len(tokens_all) == len(attention_masks), f'{len(tokens_all)} != {len(attention_masks)}'
            assert len(tokens_all) == len(attention_pred_mask), f'{len(tokens_all)} != {len(attention_pred_mask)}'
            if suffix == "":
               text_list.append(tokens_all)
               attention_list.append(attention_masks)
               attention_pred_mask_list.append(attention_pred_mask)
            else:
               text_code_list.append(tokens_all)
               attention_code_list.append(attention_masks)
               attention_code_pred_mask_list.append(attention_pred_mask)
        
    print("attention_error:",count)
    print("no_majority:",count_confused)
    # Calling DataFrame constructor after zipping 
    # both lists, with columns specified 

    Post_ids_list = data["Post_id"]
    label_list = data["label"]

    training_data = pd.DataFrame(list(zip(Post_ids_list, text_list, attention_list, label_list, attention_pred_mask_list,
                                                         text_code_list, attention_code_list, attention_code_pred_mask_list)), 
                   columns =['Post_id', 'Text', 'Attention' , 'Label', 'attention_pred_mask',
                                        'Text_code', 'Attention_code', 'attention_code_pred_mask']) 
    filename=set_name(params)
    print(filename)
    print(f'sample tokens: {text_list[2]}')
    return training_data



def convert_data(test_data_list, test_post_ids, cve_end_idx_list, params, list_dict, rational_present=True):
    """this converts the data to be with or without the rationals based on the previous predictions"""
    """input: params -- input dict, list_dict -- previous predictions containing rationals
    rational_present -- whether to keep rational only or remove them only
    topk -- selected topk_indices or top k
    
    if rational_present == True:
        only keep the rationals
    else:
        remove the rationals
    
    test_data: dataframe
    """
    tokenizer = RobertaTokenizerFast.from_pretrained(params["model_name"])

    temp_dict={}
    for ele in list_dict:
        temp_dict[ele['annotation_id']]=ele['rationales'][0] #['soft_rationale_predictions']
    
    test_data_modified = []
    test_post_ids_modified = []
    start_id, end_id = get_start_end_sep_token(params)
    
    for index in tqdm(range(len(test_data_list))):
        row = test_data_list[index]
        try:
            attention=temp_dict[test_post_ids[index]]  #为什么这里要费劲的用key，而不是index？因为temp dict只包含gt为1的example，和test data list长度是不一样的
        except KeyError:
            continue
        #topk_indices = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]
        # else:
        #     topk_indices = [i for i in range(len(attention)) if attention[i] > 0]
        new_text =[]
        new_text_attention =[]

        new_code = []
        new_code_attention = []
        row_text = row[2]
        row_text_attention = row[3]
        row_code = row[5]
        row_code_attention = row[6]

        this_cve_end_idx = cve_end_idx_list[index]

        new_cve_text = row_text[:this_cve_end_idx]
        new_cve_attention_text = row_text_attention[:this_cve_end_idx]

        new_cve_code = []
        new_cve_attention_code = []

        old_codeidx2new_codeidx = {}

        if(rational_present):  
            new_cve_text.append(start_id)
            new_cve_attention_text.append(0)

            for each_idx in sorted(attention["topk_text"]):
                new_cve_text.append(row_text[each_idx])
                new_cve_attention_text.append(row_text_attention[each_idx])

            new_cve_text.append(end_id)
            new_cve_attention_text.append(0)
            
            this_code_binidx = 0
            for each_bin in sorted(attention["topk_code"].keys()):
                new_cve_code.append(row_text[:this_cve_end_idx] + [start_id])
                new_cve_attention_code.append(row_text_attention[:this_cve_end_idx] + [0])

                for each_idx in sorted(attention["topk_code"][each_bin]):
                    new_cve_code[this_code_binidx].append(row_code[each_bin][each_idx])
                    new_cve_attention_code[this_code_binidx].append(row_code_attention[each_bin][each_idx])
                new_cve_code[this_code_binidx].append(end_id)
                new_cve_attention_code[this_code_binidx].append(0)

                old_codeidx2new_codeidx[each_bin] = this_code_binidx

                this_code_binidx += 1

            if len(new_cve_code) == 0:
                new_cve_code.append(row_text[:this_cve_end_idx] + [start_id, end_id])
                new_cve_attention_code.append(row_text_attention[:this_cve_end_idx] + [0, 0])

            if len(attention["topk_code"]) == 0:
                new_sortedidx = [0]
            else:
                new_sortedidx = [old_codeidx2new_codeidx[x] for x in row[1] if x in sorted(attention["topk_code"])] #need to translate the old code bin idx to new code bin idx since the #bins may be reduced
        else:
            new_cve_text.append(start_id)
            new_cve_attention_text.append(0)

            topk_text_tokens = [tokenizer.convert_ids_to_tokens(row_text[x]).lower() for x in attention["topk_text"]]

            for each_idx in range(this_cve_end_idx + 1, len(row_text) - 1):
                if each_idx not in attention["topk_text"]:
                #if not (tokenizer.convert_ids_to_tokens(row_text[each_idx]).lower() in topk_text_tokens):
                    new_cve_text.append(row_text[each_idx])
                    new_cve_attention_text.append(row_text_attention[each_idx])
            new_cve_text.append(end_id)
            new_cve_attention_text.append(0)

            for each_bin in range(len(row_code)):
                new_cve_code.append(row_text[:this_cve_end_idx] + [start_id])
                new_cve_attention_code.append(row_text_attention[:this_cve_end_idx] + [0])
                for each_idx in range(this_cve_end_idx + 1, len(row_code[each_bin]) - 1):
                    this_bin_top_indices = attention["topk_code"].get(each_bin, [])
                    #this_bin_top_indices_tokens = [tokenizer.convert_ids_to_tokens(row_code[each_bin][x]).lower() for x in this_bin_top_indices]

                    if each_idx not in this_bin_top_indices:
                    #if not (tokenizer.convert_ids_to_tokens(row_code[each_bin][each_idx]).lower() in this_bin_top_indices_tokens):
                        new_cve_code[each_bin].append(row_code[each_bin][each_idx])
                        new_cve_attention_code[each_bin].append(row_code_attention[each_bin][each_idx])
                new_cve_code[each_bin].append(end_id)
                new_cve_attention_code[each_bin].append(0)
            new_sortedidx = row[1]

        test_data_modified.append([row[0], new_sortedidx, new_cve_text, new_cve_attention_text, [], new_cve_code, new_cve_attention_code, []])
        test_post_ids_modified.append(test_post_ids[index])
        # else:        
        #     test_data_modified.append([row['Post_id'], new_text, new_attention, row['Label']])
    print('Data converted')
    #print(f'original text: {test_data.iloc[0]["Text"]}')
    #df = pd.DataFrame(test_data_modified, columns=['Post_id', 'Label', 'sorted_idx', 'Text', 'Attention', 'attention_pred_mask', 'Text_code', 'Attention_code', 'attention_pred_mask_code'])
    return test_data_modified, test_post_ids_modified

def get_start_end_sep_token(params):
    if "codebert" in params["model_name"] or "unixcoder" in params["model_name"]:
        return 0, 2
    if "bert" in params["model_name"]:
        return 101, 102


def collect_data_trace(params, fold):
    if (params['bert_tokens']):
        print('Loading BERT tokenizer...')
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
        model_name = params["model_name"]   # only get tokenizer
        # tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_special_tokens=True)
        #tokenizer.add_tokens(["[NEWLINE]"])
        print('use tokenizer: ', model_name, tokenizer)
    else:
        tokenizer=None
    trace_data_folder = "Data/" + set_name(params)

    if params["attfirst"] is True:
        trace_data_path = trace_data_folder + f'/output_Dataset_{fold}_ranked_attfirst.json'
    else:
        trace_data_path = trace_data_folder + f'/output_Dataset_{fold}_ranked_2.json'
    #     'valid': trace_data_folder + '/output_Dataset_valid.json',
    #     'test': trace_data_folder + '/output_Dataset_test.json'
    # }

    this_data = get_annotated_data_trace(params, trace_data_path, fold)
    return this_data


#### OLDcode remove at last
def return_inverse_dict():
    with open("../../main/id_orig_seid_Mapping.json") as f:
        id_dict_orig = json.load(f) 



    orig_dict_id={}
    for key in tqdm(id_dict_orig.keys()):    
        orig_text=id_dict_orig[key][0]
        seid_text=id_dict_orig[key][1]

        orig_dict_id[seid_text]=[key,orig_text]
    return orig_dict_id


def return_id_orig(text,orig_dict_id):
    try:
        #to return the test directly 
        return orig_dict_id[text][0],orig_dict_id[text][1]
    except:
        max_sim=0
        max_text=""
        for key in orig_dict_id.keys():
            text_id=orig_dict_id[key][0]
            orig_text=orig_dict_id[key][1]
            sim=similar(key,text)
            if(sim>max_sim):
                max_sim=sim
                max_text=key
            if(sim>0.95):
                return text_id,orig_text
        print(text,"||",max_text,"||",max_sim)
        return -1,-1

    
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
    

