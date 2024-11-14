# -*- coding: utf-8 -*-
import torch
from transformers import RobertaTokenizer,RobertaTokenizerFast
#### common utils
from Models.utils import fix_the_random,format_time,get_gpu,return_params
#### metric utils 
from Models.utils import softmax,return_params
from trace_params import set_params
#### model utils
from tqdm import tqdm
from TensorDataset.dataLoader import combine_features
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import time
import os
from sklearn.utils import class_weight
import json
from Models.bertModels_pooling import *
from Models.otherModels import *
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, tqdm_notebook
import pandas as pd
from torch.nn import LogSoftmax
from lime_trace.lime_trace import TraceExplainer
from lime_trace.utils import get_offset
import numpy as np
import re
import argparse
from Models.utils import params_2_model_save_path
from Preprocess.dataCollect import set_name, collect_data_trace, convert_data
from train import Eval_phase
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import pdb,pickle
import warnings
# warnings.simplefilter('always', FutureWarning)  # 启用所有 FutureWarning 的警告
warnings.simplefilter('ignore', FutureWarning)  # 忽略所有 FutureWarning 的警告

# trace back of skipped warning
# warnings.filterwarnings('error', category=FutureWarning)  # 将 FutureWarning 转换为异常
# 下面的异常时包版本的问题
    

def select_model(params,embeddings):
    if(params['bert_tokens']):
        if(params['what_bert']=='weighted'):
            model = SC_weighted_BERT.from_pretrained(
            params['path_files'], # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = params['num_classes'], # The number of output labels
            output_attentions = True, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            hidden_dropout_prob=params['dropout_bert'],
            params=params
            )
        else:
            print("Error in bert model name!!!!")
        return model
    else:
        text=params['model_name']
        if(text=="birnn"):
            model=BiRNN(params,embeddings)
        elif(text == "birnnatt"):
            model=BiAtt_RNN(params,embeddings,return_att=True)
        elif(text == "birnnscrat"):
            model=BiAtt_RNN(params,embeddings,return_att=True)
        elif(text == "cnn_gru"):
            model=CNN_GRU(params,embeddings)
        elif(text == "lstm_bad"):
            model=LSTM_bad(params)
        else:
            print("Error in model name!!!!")
        return model


class modelPred():
    def __init__(self, params, test_data):
        self.params = params
#         self.params["device"]='cuda'
        self.embeddings = None
        if params['problem'] != 'trace':
            raise NotImplementedError('Only on trace')
        
        print("Loading data from trace")
        self.test_data = test_data.iloc[:, 1:]
        self.test_data = list(self.test_data.itertuples(index=False, name=None))
        
        if torch.cuda.is_available() and self.params['device']=='cuda':    
            # Tell PyTorch to use the GPU.    
            self.device = torch.device("cuda")
            deviceID = get_gpu(self.params)
            torch.cuda.set_device(deviceID[0])
        elif self.params['device'] !='cuda':
            print('Since you dont want to use GPU, using the CPU instead.')
            self.device = torch.device("cpu")
        else:
            print('gpu is not available, using the CPU instead.')
            self.device = torch.device("cpu")

        if (params['auto_weights']):
            y_test = [ele[0] for ele in test_data] 
            encoder = LabelEncoder()
            encoder.classes_ = np.load(params['class_names'])
            params['weights'] = class_weight.compute_class_weight('balanced',np.unique(y_test),y_test).astype('float32')

        self.model = select_model(self.params, self.embeddings)
        
        if (self.params["device"]=='cuda'):
            self.model.cuda()
        self.model.eval()
    
    def return_probab(self, new_test_data, test_post_ids=None, batch_size=16):
        """Input: should be a list of sentences"""
        """Ouput: probablity values"""
        params = self.params
        device = self.device

        # new_test_data = new_test_data.iloc[:, 1:]
        # new_test_data = list(new_test_data.itertuples(index=False, name=None))
        
        # tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base', do_lower_case=False)
        # input_test_data = []
        # for sentence in test_data_list:
        #     token_ids = tokenizer.encode(sentence, add_special_tokens=True, truncation=False)
        #     input_test_data.append((token_ids, [1]*len(token_ids), 0, [1]*len(token_ids)))  # dummy label, and attention mask, dummy attention pred mask
        #test_post_ids = list(test_data['Post_id'])
        #raise Exception(new_test_data[1091][5])
        new_test_dataloader = combine_features(new_test_data, post_ids=test_post_ids, params=params, is_train=False, batch_size = batch_size)
        
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        # Tracking variables
        # post_id_all = list(test_data['Post_id'])

        print("Running eval on test data...")
        t0 = time.time()
        true_labels=[]
        pred_labels=[]
        logits_all=[]
        #attention_all=[]
        #input_mask_all=[]

        # Evaluate data for one epoch
        for step,batch in tqdm(enumerate(new_test_dataloader)):

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

            b_code_ids_padded = pad_sequence(batch[5], batch_first=True, padding_value=0).data.to(device)
            b_code_ids_packed = pack_padded_sequence(b_code_ids_padded, lengths=b_code_num, batch_first=True, enforce_sorted=False)
            b_code_att_padded = pad_sequence(batch[6], batch_first=True, padding_value=0).data.to(device)
            b_code_att_packed = pack_padded_sequence(b_code_att_padded, lengths=b_code_num, batch_first=True, enforce_sorted=False)
            b_code_mask_padded = pad_sequence(batch[7], batch_first=True, padding_value=0).data.to(device)
            b_code_mask_packed = pack_padded_sequence(b_code_mask_padded, lengths=b_code_num, batch_first=True, enforce_sorted=False)
            b_labels = torch.stack(batch[0]).to(device)

            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            #model.zero_grad() 

            #if test_post_ids[step] == "CVE-2017-18077/b13381281cead487cbdbfd6a69fb097ea5e456c3": # "CVE-2015-0219/4f6fffc1dc429f1ad428ecf8e6620739e8837450":
            #    import pdb; pdb.set_trace()       
            outputs = self.model(input_ids=b_text_ids, 
                text_att=b_text_att,
                text_mask=b_text_mask, 
                code_ids=b_code_ids_packed,
                code_att=b_code_att_packed,
                code_mask=b_code_mask_packed,
                code_lengths=b_code_num,
                labels=None,
                device=device)
            
            logits = outputs[0]
            #print(logits)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.detach().cpu().numpy()

            # Calculate the accuracy for this batch of test sentences.
            # Accumulate the total accuracy.
            pred_labels+=list(np.argmax(logits, axis=1).flatten())
            true_labels+=list(label_ids.flatten())
            logits_all+=list(logits)
            #attention_all+=list(attention_vectors)
            #input_mask_all+=list(batch[2].detach().cpu().numpy())

        logits_all_final=[]
        for logits in logits_all:
            logits_all_final.append(list(softmax(logits)))

        return np.array(logits_all_final)

def replace_tokens(tokens):
    return [x.replace("\u0120", " ").replace("\u010A", "\n").replace("\u0109", "\t").encode('latin1', errors='ignore').decode('utf-8', errors='ignore') for x in tokens]

def standaloneEval_with_lime(params, test_data_list=None, test_post_ids=None, modelClass=None, topk=2,rational=False,highlight_mode = "text", postid_to_len = {}, filter_mode="none"):
    from explain_with_tfidf import should_filter
    if filter_mode == "manual":
        manual_postid_list = list(pd.read_csv("./print/sampled_postids.csv", sep=",")["postids"])
    else:
        manual_postid_list = None

    if params['problem'] != 'trace':
        raise NotImplementedError('Only on trace')
    tokenizer = RobertaTokenizerFast.from_pretrained(params['model_name'])

    encoder = LabelEncoder()
    encoder.classes_ = np.load(params['class_names'])
    explainer = TraceExplainer(class_names=list(encoder.classes_),random_state=333,bow=False)

    list_dict=[]

    if (rational==True):
        sentence_list=[]
        post_id_list=[]

        assert type(test_data_list) == list
        assert len(test_data_list) == len(test_post_ids)

        #raise Exception(len(test_data_list), test_data_list[1091][5])
        
        probab_list = modelClass.return_probab(test_data_list, test_post_ids = test_post_ids, batch_size=1)
        assert len(test_post_ids) == len(probab_list)  == len(test_data_list)

        for post_id,proba, this_data_entry in zip(test_post_ids, list(probab_list), test_data_list):
            if post_id in ["CVE-2021-32803/46fe35083e2676e31c4e0a81639dce6da7aaa356", "CVE-2022-24799/d14455252a949dc83f36d45e2babbdd9328af2a4","CVE-2011-3371/dd50a50a2760f10bd2d09814e30af4b36052ca6d", "CVE-2021-32643/52e1890665410b4385e37b96bc49c5e3c708e4e9","CVE-2021-21272/96cd90423303f1bb42bd043cb4c36085e6e91e8e", "CVE-2021-41125/b01d69a1bf48060daec8f751368622352d8b85a6", "CVE-2022-24713/ae70b41d4f46641dbc45c7a4f87954aea356283e"]: continue
            temp={}
            temp["this_true_label"] = this_data_entry[0]
            temp["annotation_id"] = post_id
            temp["classification_scores"] = {"non-patch":proba[0], "patch":proba[1]}
            list_dict.append(temp)

        return list_dict
    else:
        cve_end_idx_list = []
        test_dataloader = combine_features(test_data_list, params, is_train=False, batch_size=1)

        true_labels = []
        pred_scores = []

        top_labels = 2
        num_samples = params['num_samples']

        start_time = time.time()

        # Must use the dataloader because the tokens were truncated in the roberta model
        for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            if step > 33000: continue
            temp={}
            b_labels = batch[0][0]

            sorted_idx = test_data_list[step][1]

            b_text_ids = torch.stack(batch[2])[0,:]
            b_text_att = torch.stack(batch[3])[0,:]
            b_text_mask = torch.stack(batch[4])[0,:]
            b_code_ids = torch.stack(batch[5])[0,:]
            b_code_att = torch.stack(batch[6])[0,:]
            b_code_mask = torch.stack(batch[7])[0,:]
            b_post_ids = batch[1]

            cve_desc_end_idx_ = [x for x in range(1, len(b_text_mask)) if b_text_mask[x] == 0]#[0]
            if len(cve_desc_end_idx_) > 0:
                cve_desc_end_idx = cve_desc_end_idx_[0]
            else:
                cve_desc_end_idx = len(b_text_mask)
            cve_end_idx_list.append(cve_desc_end_idx)
            #if int(b_labels) != 1: continue
            #if should_filter(test_post_ids[step], int(b_labels), filter_mode, manual_postid_list): continue

            text_len_ = [x for x in range(1, len(b_text_mask)) if b_text_mask[x] == 0]
            if len(text_len_) > 1:
                text_len = text_len_[1]
            else:
                text_len = len(b_text_mask)
            
            b_text_ids = b_text_ids[:text_len]

            code_lens = []
            for x in range(len(b_code_ids)):
                this_code_len_ = [y for y in range(1, len(b_code_ids[x])) if b_code_mask[x][y] == 0]
                if len(this_code_len_) > 1:
                    this_code_len = this_code_len_[1]
                else:
                    this_code_len = len(b_code_ids[x]) 
                code_lens.append(this_code_len)
            b_code_ids = [b_code_ids[x][:code_lens[x]] for x in range(len(code_lens))]

            assert len(b_code_ids) == 1

            this_row = test_data_list[step]

            if test_post_ids[step] in ["CVE-2021-32803/46fe35083e2676e31c4e0a81639dce6da7aaa356", "CVE-2022-24799/d14455252a949dc83f36d45e2babbdd9328af2a4","CVE-2011-3371/dd50a50a2760f10bd2d09814e30af4b36052ca6d", "CVE-2021-32643/52e1890665410b4385e37b96bc49c5e3c708e4e9","CVE-2021-21272/96cd90423303f1bb42bd043cb4c36085e6e91e8e", "CVE-2021-41125/b01d69a1bf48060daec8f751368622352d8b85a6", "CVE-2022-24713/ae70b41d4f46641dbc45c7a4f87954aea356283e"]: continue
            if text_len - cve_desc_end_idx - 2 <= 0: continue
            if code_lens[0] - cve_desc_end_idx - 2 <= 0: continue

            exp = explainer.explain_instance(mode=highlight_mode, this_row=this_row, b_text_ids=b_text_ids, b_code_ids=b_code_ids, sorted_idx=sorted_idx, cve_desc_end_idx=cve_desc_end_idx, text_len=text_len, code_lens=code_lens, classifier_fn=modelClass.return_probab, num_features=30, top_labels=top_labels, num_samples=num_samples)
            pred_id=np.argmax(exp.predict_proba)
            pred_label=encoder.inverse_transform([pred_id])[0]
            #ground_label=row['Label']
            true_labels.append(b_labels)
            pred_scores.append(exp.predict_proba[1])

            temp["annotation_id"] = test_post_ids[step]
            temp["classification"]= ['non-patch', 'patch'][pred_label]
            temp["classification_scores"]={"non-patch":exp.predict_proba[0],"patch":exp.predict_proba[1]}

            attention_text = [0] * len(this_row[2])
            attention_code = [[0] * len(each_code_ids) for each_code_ids in this_row[5]]
            #attention_pred_mask = row['attention_pred_mask']
            explanation = sorted(exp.as_map()[1], key = lambda x:x[1], reverse=True)#[:postid_to_len[test_post_ids[step]]]

            temp_hard_rationales_text = []
            temp_hard_rationales_code = []
            for idx in range(len(this_row[5])):
                temp_hard_rationales_code.append([])

            list_of_explanations_text = []
            list_of_explanations_code = {}

            num_of_tokens = postid_to_len[test_post_ids[step]]
            this_all_tokens = set([])
            
            for each_exp in explanation:
                idx, score = each_exp[0], each_exp[1]

                if highlight_mode == "textonly":
                    bin_id = -1
                    component = "text"
                    offset = idx
                elif highlight_mode == "codeonly":
                    bin_id = sorted_idx[0]
                    component = "code"
                    offset = idx
                else:
                    component, bin_id, offset = get_offset(idx, exp.sizes_cumsum, this_row)

                if score > 0:
                    if component == "text":
                        this_token = replace_tokens(tokenizer.convert_ids_to_tokens([b_text_ids[cve_desc_end_idx + 1 + offset]]))[0]
                        if not re.search("[a-zA-Z0-9]", this_token): continue
                        if this_token.lower() in this_all_tokens: continue
                        if len(list_of_explanations_text) >= num_of_tokens: continue
                        this_all_tokens.add(this_token.lower())
                        attention_text[offset + cve_desc_end_idx + 1] = score
                        temp_hard_rationales_text.append({'end_token': offset + cve_desc_end_idx + 2, 'start_token': offset + cve_desc_end_idx + 1})
                        list_of_explanations_text.append(cve_desc_end_idx + 1 + offset)
                    else:
                        this_token = replace_tokens(tokenizer.convert_ids_to_tokens([b_code_ids[0][cve_desc_end_idx + 1 + offset]]))[0]
                        #import pdb; pdb.set_trace()
                        if not re.search("[a-zA-Z0-9]", this_token): continue
                        if this_token.lower() in this_all_tokens: continue
                        list_of_explanations_code.setdefault(bin_id, [])
                        if len(list_of_explanations_code[bin_id]) >= num_of_tokens: continue
                        this_all_tokens.add(this_token.lower())
                        # if code, need to check whether the token is special char only or repeated
                        attention_code[bin_id][offset + cve_desc_end_idx + 1] = score
                        temp_hard_rationales_code[bin_id].append({'end_token': offset + cve_desc_end_idx + 2, 'start_token': offset + cve_desc_end_idx + 1})
                        list_of_explanations_code[bin_id].append(cve_desc_end_idx + 1 + offset)
            # if (rational==False):
            #     assert(len(attention) == len(row['Attention']))
            #if len(list_of_explanations_code) > 0:
            #    print("ttt", len(list(list_of_explanations_code.values())[0]), "sss")
            #if len(list_of_explanations_text) > 0:
            #    print("ttt", len(list_of_explanations_text), "sss")


            temp["rationales"] = [{"docid": test_post_ids[step], 
                                     "hard_rationale_predictions_text": temp_hard_rationales_text, 
                                     "hard_rationale_predictions_code": temp_hard_rationales_code,
                                     "soft_rationale_predictions_text": attention_text,
                                     "soft_rationale_predictions_code": attention_code,
                                    #  "text": replace_tokens(tokenizer.convert_ids_to_tokens(this_row[2])),
                                    #  "code": [replace_tokens(tokenizer.convert_ids_to_tokens(each_code_ids)) for each_code_ids in this_row[5]],
                                    "topk_text": list_of_explanations_text,
                                    "topk_code": list_of_explanations_code,
                                    "cve_desc_end_idx": cve_desc_end_idx,
                                     "truth": int(b_labels)}]  # i dont understand
            list_dict.append(temp)

        print("time cost: ", time.time() - start_time)
                
        return list_dict, true_labels, pred_scores, cve_end_idx_list


def get_final_dict_with_lime(params, test_data, topk, stage = 0, highlight_mode = "textonly", filter_mode="none", fold="valid", topk_2=2):
    # from rank_metrics import compute_metrics

    modelClass = modelPred(params, test_data)

    test_post_ids = list(test_data['Post_id'])

    param_2_path = params_2_model_save_path(params)
    param_2_path = 'explanations_dicts/' + param_2_path  #.replace('Saved/', 'explanations_dicts/')
    if not os.path.exists('explanations_dicts'):
        os.makedirs('explanations_dicts')
    if not os.path.exists(param_2_path):
        os.makedirs(param_2_path)

    list_dict_org_path = os.path.join(param_2_path, f'list_dict_org_{highlight_mode}_{fold}.pkl')

    postid_to_len = {}
    if highlight_mode == "textonly":
        gt_output_path = os.path.join(param_2_path, f'explanation_with_tfidf_textonly_{fold}.json')
        gt_results = [json.loads(line) for line in open(gt_output_path, "r").readlines()]
        postid_to_len = {entry["annotation_id"]: len(entry["rationales"][0]["topk_text"]) for entry in gt_results}
    else:
        gt_output_path = os.path.join(param_2_path, f'explanation_with_tfidf_codeonly_{fold}.json')
        gt_results = [json.loads(line) for line in open(gt_output_path, "r").readlines()]
        postid_to_len = {entry["annotation_id"]: len(list(entry["rationales"][0]["topk_code"].values())[0]) for entry in gt_results}

    test_data = test_data.iloc[:, 1:]
    test_data_list = list(test_data.itertuples(index=False, name=None))

    if stage == 0:
        list_dict_org, true_labels, pred_scores, cve_end_idx_list = standaloneEval_with_lime(params, test_data_list=test_data_list, test_post_ids=test_post_ids, modelClass=modelClass, topk=topk, rational = False, highlight_mode=highlight_mode, postid_to_len=postid_to_len, filter_mode = filter_mode)
        # metric_dict = compute_metrics(pred_scores, true_labels, list(test_data["Post_id"]))    
        pickle.dump({"cve_end_idx_list": cve_end_idx_list, "list_dict_org": list_dict_org}, open(list_dict_org_path, "wb"))
    else:
        pickle_data = pickle.load(open(list_dict_org_path, "rb"))
        list_dict_org = pickle_data["list_dict_org"]
        cve_end_idx_list = pickle_data["cve_end_idx_list"]

        for x in range(len(list_dict_org)):
            list_dict_org[x]["rationales"][0]["topk_text"] = list_dict_org[x]["rationales"][0]["topk_text"][:topk_2]
            for y in list(list_dict_org[x]["rationales"][0]["topk_code"].keys()):
                list_dict_org[x]["rationales"][0]["topk_code"][y] = list_dict_org[x]["rationales"][0]["topk_code"][y][:topk_2]

        test_data_with_rational, test_post_ids_modified = convert_data(test_data_list=test_data_list, test_post_ids=test_post_ids, cve_end_idx_list=cve_end_idx_list, params=params, list_dict=list_dict_org,rational_present=True)

        list_dict_with_rational = standaloneEval_with_lime(params, test_data_list=test_data_with_rational, test_post_ids=test_post_ids_modified, modelClass=modelClass, topk=topk,rational=True)

        test_data_without_rational, test_post_ids_modified = convert_data(test_data_list=test_data_list, test_post_ids=test_post_ids, cve_end_idx_list=cve_end_idx_list, params=params, list_dict=list_dict_org, rational_present=False)
        
        list_dict_without_rational = standaloneEval_with_lime(params, test_data_list=test_data_without_rational, test_post_ids = test_post_ids_modified, modelClass = modelClass, topk=topk, rational=True)

        assert len(list_dict_org) == len(list_dict_with_rational) == len(list_dict_without_rational)

        final_list_dict=[]
        for ele1,ele2,ele3 in zip(list_dict_org, list_dict_with_rational, list_dict_without_rational):
            ele1['sufficiency_classification_scores']=ele2['classification_scores']
            ele1['comprehensiveness_classification_scores']=ele3['classification_scores']
            final_list_dict.append(ele1)
        
        path_name_explanation = os.path.join(param_2_path, f'explanation_with_lime_{highlight_mode}_{fold}.json') #_{topk_2}.json')  
        with open(path_name_explanation, 'w') as fp:
            for i in final_list_dict:
                fp.write(json.dumps(i, cls=NumpyEncoder) + "\n")
        print(path_name_explanation)
        # print("metric dict:", metric_dict)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def add_cve_desc_highlight(params, test_data):
    param_2_path = params_2_model_save_path(params)
    param_2_path = 'explanations_dicts/' + param_2_path #.replace('Saved/', 'explanations_dicts/')
    if not os.path.exists('explanations_dicts'):
        os.makedirs('explanations_dicts')
    if not os.path.exists(param_2_path):
        os.makedirs(param_2_path)

    explanation_path_input = os.path.join(param_2_path, f'explanation_with_lime.json')
    explanation_path_output = os.path.join(param_2_path, f'explanation_with_lime_full_highlight.json')

    tokenizer = RobertaTokenizerFast.from_pretrained(params['model_name'])

    with open(explanation_path_input, "r") as fin:
        lines = fin.readlines()
        fout = open(explanation_path_output, "w")
        for idx in tqdm(range(len(lines))):
            this_entry = json.loads(lines[idx])
            this_entry_soft_text = this_entry["rationales"][0]["soft_rationale_predictions_text"]
            this_entry_soft_code = this_entry["rationales"][0]["soft_rationale_predictions_code"]

            this_text = test_data["Text"][idx]
            this_code = test_data["Text_code"][idx]

            cve_desc_end_idx_ = [x for x in range(1, len(this_text)) if this_text[x] == 0]
            if len(cve_desc_end_idx_) == 0:
                cve_desc_end_idx = len(this_text)
            else:
                cve_desc_end_idx = cve_desc_end_idx_[0]
            
            all_tokens = {} 
            for x in range(len(this_entry_soft_text)):
                if this_entry_soft_text[x] > 0:
                    this_token = tokenizer.convert_ids_to_tokens(this_text[x]).lower()
                    all_tokens.setdefault(this_token, -1)
                    all_tokens[this_token] = max(all_tokens[this_token], this_entry_soft_text[x])
            for x in range(len(this_entry_soft_code)):
                for y in range(len(this_entry_soft_code[x])):
                    if this_entry_soft_code[x][y] > 0:
                        this_token = tokenizer.convert_ids_to_tokens(this_code[x][y]).lower()
                        all_tokens.setdefault(this_token, -1)
                        all_tokens[this_token] = max(all_tokens[this_token], this_entry_soft_code[x][y])
            this_entry["rationales"][0]["highlighted_tokens"] = all_tokens
            cve_highlighted_index = [x for x in range(cve_desc_end_idx) if tokenizer.convert_ids_to_tokens(this_text[x]).lower() in all_tokens]
            this_entry["rationales"][0]["topk_cve"] = cve_highlighted_index
            fout.write(json.dumps(this_entry) + "\n")
        fout.close()
    
if __name__=='__main__': 

    my_parser = set_params()
    my_parser.add_argument("--fold", type=str, required=False)
    my_parser.add_argument("--highlight_mode", type=str, required=False)
    my_parser.add_argument("--topk", type=str, required=False)

    args = my_parser.parse_args()
    args.att_lambda = 0

    # directly give the the json file as parameter
    params = return_params(args.path, args, load_trained_model=True)
    assert 'num_samples' in params
    
    if params['problem'] != 'trace':
        # raise NotImplementedError('Only on trace')
        pass

    #raise Exception(params_2_model_save_path(params))
        
    # load the trace data
    # load pickled data
    fold = args.fold
    stage = 0
    highlight_mode = args.highlight_mode
    filter_mode = "none"
    test = collect_data_trace(params, fold=fold)
    topk =args.topk
    topk_2 = 10000

    #if part is not None:
    #for highlight_mode in ["codeonly"]:
    get_final_dict_with_lime(params, test, topk=topk, stage= 0, highlight_mode=highlight_mode, filter_mode=filter_mode, fold=fold) # text:4 because the average number of tokens in tf idf highlighting is 3.87; code: 7 because the average number of tokens in tf idf highlighting is 6.79
    #    for topk_2 in [3, 5, 10, 20, 30]:
    get_final_dict_with_lime(params, test, topk=topk, stage= 1, highlight_mode=highlight_mode, filter_mode=filter_mode, fold=fold, topk_2 = topk_2) # text:4 because the average number of tokens in tf idf highlighting is 3.87; code: 7 because the average number of tokens in tf idf highlighting is 6.79

    #else:
    #add_cve_desc_highlight(params, test_data=test)

    # reverse match the cve description tokens
    
