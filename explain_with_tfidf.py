import json, torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from explain_with_lime import replace_tokens
from Preprocess.dataCollect import set_name, collect_data_trace, convert_data
from TensorDataset.dataLoader import combine_features
from transformers import RobertaTokenizer,RobertaTokenizerFast
import tqdm
import numpy as np
from rank_metrics import compute_metrics
from Models.utils import return_params
from torch.nn.functional import softmax
from trace_params import set_params
import traceback
from explain_with_lime import standaloneEval_with_lime,modelPred,NumpyEncoder
from Models.utils import params_2_model_save_path
import os
import sys
import re
import pandas

# paths = ["Trace_microsoft-codebert-base_lam=0.005_maxlen=512_k=1_beta=0.8_epoch=4", "Trace_microsoft-codebert-base_lam=0_maxlen=256_k=1_epoch=5_pooling=cnn_attfirst=True"]

def get_data(path, suffix, fold):
    if path != "":
        file_path = "./explanations_dicts/" + path + f"/explanation_with{suffix}_{fold}.json"
    else:
        file_path = "./explanations_dicts/explanation_with_lime.json"
    with open(file_path, "r") as fin:
        lines = fin.readlines()
        data = [json.loads(line) for line in lines]
        return data

def get_prec_w_highlight(data, test_post_ids, option):
    proba_list_suff = [entry['sufficiency_classification_scores']["patch"] for entry in data]
    proba_list_comp = [entry['comprehensiveness_classification_scores']["patch"] for entry in data]
    if option == "tfidf":
       y_test = [entry["this_true_label"] for entry in data]
    else:
       y_test = [entry["rationales"][0]["truth"] for entry in data]
    print("sufficiency")
    print(compute_metrics(scores=proba_list_suff, labels=y_test, topk=1, cves=test_post_ids))
    print("comprehensive")
    print(compute_metrics(scores=proba_list_comp, labels=y_test, topk=1, cves=test_post_ids))


def get_sufficiency_scores(data):
    #    scores_list = [entry["classification_scores"]["patch"] for entry in data]
    #    sufficiency_scores_list = [entry["sufficiency_classification_scores"]["patch"] for entry in data]
    #    truth_list = [entry["rationales"][0]["truth"] for entry in data]
    #    cves = [entry["annotation_id"].split("/")[0] for entry in data]
       avg_sufficiency = [entry['sufficiency_classification_scores']["patch"] for entry in data]
       return  np.mean(avg_sufficiency)

def get_comprehensive_scores(data):
    #    scores_list = [entry["classification_scores"]["patch"] for entry in data]
    #    comprehensive_scores_list = [entry["comprehensiveness_classification_scores"]["patch"] for entry in data]
    #    truth_list = [entry["rationales"][0]["truth"] for entry in data]
    #    cves = [entry["annotation_id"].split("/")[0] for entry in data]
    #    import pdb; pdb.set_trace()
    #    print(compute_metrics(scores=scores_list, labels=truth_list, topk=1, cves=cves))
       avg_comprehensive = [entry['comprehensiveness_classification_scores']["patch"] for entry in data]
       return np.mean(avg_comprehensive)

def get_highlighted_text(b_text_tokens, b_code_tokens, method_id, datas):
    highlighted_text = [b_text_tokens[x] for x in datas[method_id][step]["rationales"][0]["topk_text"]]
    highlighted_code = [b_code_tokens[x] for x in datas[method_id][step]["rationales"][0]["topk_code"].get("0", [])]
    return highlighted_text, highlighted_code

def compute_faithfulness_tfidf(option, fold, filter_mode, params, suffix, param_2_path, test_data_list, test_post_ids, cve_end_idx_list, modelClass, id2text_codetopk=None, topk=10):
    global manual_postid_list
    list_dict_org = []
    filtered_labels = []

    for idx in range(len(test_data_list)):
        list_dict = {}
        this_label = test_data_list[idx][0]
        annotation_id = test_post_ids[idx]
        #if should_filter(annotation_id, this_label, filter_mode, manual_postid_list): continue
        if annotation_id in ["CVE-2021-32803/46fe35083e2676e31c4e0a81639dce6da7aaa356", "CVE-2022-24799/d14455252a949dc83f36d45e2babbdd9328af2a4","CVE-2011-3371/dd50a50a2760f10bd2d09814e30af4b36052ca6d", "CVE-2021-32643/52e1890665410b4385e37b96bc49c5e3c708e4e9","CVE-2021-21272/96cd90423303f1bb42bd043cb4c36085e6e91e8e", "CVE-2021-41125/b01d69a1bf48060daec8f751368622352d8b85a6", "CVE-2022-24713/ae70b41d4f46641dbc45c7a4f87954aea356283e"]: continue
        if annotation_id not in id2text_codetopk: continue

        list_dict = {"annotation_id": annotation_id, "rationales": [{}]}
        list_dict["rationales"][0]["topk_text"] = id2text_codetopk.get(annotation_id, {}).get("text", [])
        list_dict["rationales"][0]["topk_code"] =  id2text_codetopk.get(annotation_id, {}).get("code", {})
        list_dict_org.append(list_dict)
        filtered_labels.append(this_label)

    test_data_with_rational, test_post_ids_modified = convert_data(test_data_list=test_data_list, test_post_ids=test_post_ids, cve_end_idx_list=cve_end_idx_list, params=params, list_dict=list_dict_org,rational_present=True)

    list_dict_with_rational = standaloneEval_with_lime(params, test_data_list=test_data_with_rational, test_post_ids=test_post_ids_modified, modelClass=modelClass, topk=topk,rational=True)

    test_data_without_rational, test_post_ids_modified = convert_data(test_data_list=test_data_list, test_post_ids=test_post_ids, cve_end_idx_list=cve_end_idx_list, params=params, list_dict=list_dict_org, rational_present=False)
    list_dict_without_rational = standaloneEval_with_lime(params, test_data_list=test_data_without_rational, test_post_ids = test_post_ids_modified, modelClass = modelClass, topk=topk, rational=True)

    assert len(list_dict_org) == len(list_dict_with_rational) == len(list_dict_without_rational) == len(filtered_labels)

    final_list_dict=[]
    for ele1,ele2,ele3, this_label in zip(list_dict_org, list_dict_with_rational, list_dict_without_rational, filtered_labels):
        ele1['sufficiency_classification_scores']=ele2['classification_scores']
        ele1['comprehensiveness_classification_scores']=ele3['classification_scores']
        ele1["this_true_label"] = this_label
        final_list_dict.append(ele1)
    
    path_name_explanation = os.path.join(param_2_path, f'explanation_with{suffix}_{fold}.json')  
    with open(path_name_explanation, 'w') as fp:
        for i in final_list_dict:
            fp.write(json.dumps(i, cls=NumpyEncoder) + "\n")


def debug(test_dataloader, test_data_list, params, highlight_mode, fold, topk, test_post_ids):

    param_2_path = 'explanations_dicts/' + params_2_model_save_path(params) 
    suffix = highlight_mode[:-4]

    lines = open(os.path.join(param_2_path, f"explanation_with_lime_{highlight_mode}_{fold}.json"), "r").readlines()
    score_lime = [json.loads(line)["sufficiency_classification_scores"]["patch"] for line in lines]
    score_lime_comp = [json.loads(line)["comprehensiveness_classification_scores"]["patch"] for line in lines]
    #cve2score = json.load(open("explanations_dicts/" + params["model_name"].replace("/", "-") + "_" + fold + ".json"))
    scores_all = []
    filtered_post_ids = []
    for line in lines:
        entry = json.loads(line)
        scores_all.append(entry["classification_scores"]["patch"])
        filtered_post_ids.append(entry["annotation_id"])

    token_lime = [] #json.loads(line)["rationales"][0][f"topk_{suffix}"] for line in lines] 
    y_test = []
    tokenlimedict = {}
    lime_len = []
    for line in lines:
        entry = json.loads(line)
        y_test.append(entry["rationales"][0]["truth"])
        commit = entry["annotation_id"]
        token_lime.append((commit, entry["rationales"][0][f"topk_{suffix}"], entry["sufficiency_classification_scores"]["patch"]))
        tokenlimedict[commit] = (entry["rationales"][0][f"topk_{suffix}"], entry["sufficiency_classification_scores"]["patch"])
        if suffix == "text":
          lime_len.append(len(entry["rationales"][0][f"topk_{suffix}"]))
        else:
          value =entry["rationales"][0][f"topk_{suffix}"].values()
          if len(value) > 0:
              lime_len.append(len(list(value)[0]))
          else:
              lime_len.append(0)
    #import pdb; pdb.set_trace()
    if highlight_mode == "textonly":
       token_lime_len = [len(x[1]) for x in token_lime]
    else:
       token_lime_len = [len(list(x[1].values())[0]) if len(x[1].values()) > 0 else 0 for x in token_lime]

    lines = open(os.path.join(param_2_path, f"explanation_with_gt_{highlight_mode}_{fold}.json"), "r").readlines()
    score_gt = [json.loads(line)["sufficiency_classification_scores"]["patch"] for line in lines] 
    score_gt_comp = [json.loads(line)["comprehensiveness_classification_scores"]["patch"] for line in lines] 


    token_gt = []
    tokengtdict = {}
    gt_len = []
    for line in lines:
        entry = json.loads(line)
        commit = entry["annotation_id"]
        token_gt.append((commit, entry["rationales"][0][f"topk_{suffix}"], entry["sufficiency_classification_scores"]["patch"])) # for line in lines]
        tokengtdict[commit] = (entry["rationales"][0][f"topk_{suffix}"], entry["sufficiency_classification_scores"]["patch"])
        if suffix == "text":
            gt_len.append(len(entry["rationales"][0][f"topk_{suffix}"]))
        else:
            value = entry["rationales"][0][f"topk_{suffix}"].values()
            if len(value) > 0:
                gt_len.append(len(list(value)[0]))
            else:
                gt_len.append(0)
    if highlight_mode == "codeonly":
       token_gt_len = [len(list(x[1].values())[0]) if len(x[1].values()) > 0 else 0 for x in token_gt]
    else:
       token_gt_len = [len(x[1]) for x in token_gt]
    largest_gap_idx = [y[0] for y in sorted([(token_lime[x][0], token_lime[x][2] - token_gt[x][2]) for x in range(len(token_lime))], key = lambda x:x[1], reverse=True)[0:10]]
    print("tfidf", np.mean(gt_len))
    print("lime", np.mean(lime_len))
    from scipy.stats import ttest_rel
    assert len(score_gt) == len(score_lime) == len(y_test) == len(filtered_post_ids) == len(scores_all)
    prec_at_kl_gt_suff = compute_metrics(scores=score_gt, labels=y_test, topk=1, cves=filtered_post_ids)["precision@kl"]
    prec_at_kl_lime_suff = compute_metrics(scores=score_lime, labels=y_test, topk=1, cves=filtered_post_ids)["precision@kl"]
    prec_at_kl_gt_comp = compute_metrics(scores=score_gt_comp, labels=y_test, topk=1, cves=filtered_post_ids)["precision@kl"]
    prec_at_kl_lime_comp = compute_metrics(scores=score_lime_comp, labels=y_test, topk=1, cves=filtered_post_ids)["precision@kl"]
    prec_at_kl_all = compute_metrics(scores=scores_all, labels=y_test, topk=1, cves=filtered_post_ids)["precision@kl"]
    length = len(prec_at_kl_all)
    import pdb; pdb.set_trace()

    gt_suff_score = [np.absolute(prec_at_kl_all[x] - prec_at_kl_gt_suff[x]) for x in range(length)]
    lime_suff_score = [np.absolute(prec_at_kl_all[x] - prec_at_kl_lime_suff[x]) for x in range(length)]
    gt_comp_score = [np.absolute(prec_at_kl_all[x] - prec_at_kl_gt_comp[x]) for x in range(length)]
    lime_comp_score = [np.absolute(prec_at_kl_all[x] - prec_at_kl_lime_comp[x]) for x in range(length)]

    print(np.mean(gt_suff_score), np.mean(lime_suff_score), ttest_rel(gt_suff_score, lime_suff_score))
    print(np.mean(gt_comp_score), np.mean(lime_comp_score), ttest_rel(gt_comp_score, lime_comp_score)) #, np.mean(score_gt), np.mean(score_lime), np.mean(score_gt_comp), np.mean(score_lime_comp))

    import pdb; pdb.set_trace()
    #return np.mean(gt_len), np.mean(score_gt), np.mean(score_gt_comp)
    return np.mean(lime_len), np.mean(score_lime), np.mean(score_lime_comp)
    

    pos_label_count = 0

    for step, batch in tqdm.tqdm(enumerate(test_dataloader)):
        label = torch.stack(batch[0])  

        this_post_id = test_post_ids[step]
        if label != 1: continue
        b_code_ids = batch[5][0][0].to(params["device"])
        original_tokens = replace_tokens(tokenizer.convert_ids_to_tokens(b_code_ids))

        this_text_ids = batch[2][0].to(params["device"])
        original_text_tokens = replace_tokens(tokenizer.convert_ids_to_tokens(this_text_ids))

        # lime_highlight_tokens = [original_tokens[x] for x in list(token_lime[pos_label_count].values())[0]]
        # gt_highlight_tokens = [original_tokens[x] for x in list(token_gt[pos_label_count].values())[0]]

        #if this_post_id == "CVE-2015-0219/4f6fffc1dc429f1ad428ecf8e6620739e8837450":
        if this_post_id in largest_gap_idx:
            import pdb; pdb.set_trace()
        pos_label_count += 1
        # highlighted_tokens = replace_tokens(tokenizer.convert_ids_to_tokens([b_text_ids[x] for x in range(len(b_text_ids)) if b_text_att[x] != 0]))

def should_filter(this_post_id, this_label, filter_mode, manual_postid_list = None):
    if filter_mode == "pos":
        return this_label != 1
    elif filter_mode == "manual":
        return this_post_id not in manual_postid_list

if __name__=='__main__': 

    parser = set_params()
    parser.add_argument("--option", type=str, required=False)
    parser.add_argument("--topk", type=int, required=False)
    parser.add_argument("--highlight_mode", type=str, required=False) 
    parser.add_argument("--fold", type=str, required=False) 

    args = parser.parse_args()
    args.att_lambda = 0

    global manual_postid_list

    option = args.option #"tfidf" # option = reproduce: reproducing the sufficieny and comprehensiveness score in testing_with_lime; option = gt: use the ground truth to compute the sufficiency and comprehensiveness; option = "lime": get the sufficiency/comprehensive score for lime
    topk = args.topk
    highlight_mode = args.highlight_mode #"textonly"
    filter_mode = "pos" # pos: use all positive example, manual: get the result for manual labeling
    fold = args.fold #"valid"
    params = return_params(args.path, args, load_trained_model=True)

    if filter_mode == "manual":
        manual_postid_list = list(pandas.read_csv("./print/sampled_postids.csv", sep=",")["postids"])

    tokenizer = RobertaTokenizerFast.from_pretrained(params['model_name'])
    from sklearn.utils import class_weight

    # directly give the the json file as parameter
    test_data = collect_data_trace(params, fold=fold)
    test_post_ids = list(test_data['Post_id'])
    test_data = test_data.iloc[:, 1:]
    test_data_list = list(test_data.itertuples(index=False, name=None))
    test_dataloader = combine_features(test_data_list, params, is_train=False, batch_size=1)
    cve_end_idx_list = []

    y_test = [entry[0] for entry in test_data_list]

    path = params_2_model_save_path(params)
    params['path_files'] = path
    params['weights'] = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_test), y=y_test).astype('float32')  
    #params["beta"] = 0.8 if path_idx == 0 else 0.7
    device = params["device"]
    model_name_hyphen = params["model_name"].replace("/", "-")

    for step, batch in tqdm.tqdm(enumerate(test_dataloader)):
        b_text_mask = torch.stack(batch[4]).to(device)[0]

        cve_desc_end_idx_ = [x for x in range(1, len(b_text_mask)) if b_text_mask[x] == 0]
        if len(cve_desc_end_idx_) > 0:
            cve_desc_end_idx = cve_desc_end_idx_[0]
        else:
            cve_desc_end_idx = len(b_text_mask)
        cve_end_idx_list.append(cve_desc_end_idx)
    #json.dump(cve_end_idx_list, open(f"cve_end_idx_list_{model_name_hyphen}_{fold}.json", "w"))
    #raise Exception("ttt")
    #cve_end_idx_list = json.load(open(f"cve_end_idx_list_{model_name_hyphen}_{fold}.json", "r"))

    param_2_path = params_2_model_save_path(params)

    param_2_path = 'explanations_dicts/' + param_2_path #.replace('Saved/', 'explanations_dicts/')
    if not os.path.exists('explanations_dicts'):
        os.makedirs('explanations_dicts')
    if not os.path.exists(param_2_path):
        os.makedirs(param_2_path)

    id2text_codetopk = {}
    for step, batch in tqdm.tqdm(enumerate(test_dataloader)):
        label = torch.stack(batch[0])  
        this_cve_desc_end_idx = cve_end_idx_list[step]

        this_post_id = test_post_ids[step]
        #if label != 1: continue
        #if should_filter(this_post_id, label, filter_mode, manual_postid_list): continue
        b_sorted_idx = test_data_list[step][1]
        b_text_mask = batch[4][0].to(device)
        b_code_mask = batch[7][0][0].to(device)

        b_text_ids = batch[2][0].to(device)
        b_text_att = batch[3][0].to(device)
        b_code_num = [len(batch[5][x]) for x in range(len(batch[5]))]

        assert len(batch[5]) == 1

        b_code_ids = batch[5][0][0].to(device)
        b_code_att = batch[6][0][0].to(device)

        b_code_tokens = replace_tokens(tokenizer.convert_ids_to_tokens(b_code_ids))
        b_text_tokens = tokenizer.convert_ids_to_tokens(b_text_ids)
        #import pdb; pdb.set_trace()
        this_code_len_ = [y for y in range(1, len(b_code_ids)) if b_code_mask[y] == 0]
        if len(this_code_len_) > 1:
            this_code_len = this_code_len_[1]
        else:
            this_code_len = len(b_code_ids)
        text_len_ = [x for x in range(1, len(b_text_mask)) if b_text_mask[x] == 0]
        if len(text_len_) > 1:
            text_len = text_len_[1]
        else:
            text_len = len(b_text_mask)
        if text_len - this_cve_desc_end_idx - 2 <= 0: continue 
        if this_code_len - this_cve_desc_end_idx - 2 <= 0: continue

        id2text_codetopk.setdefault(this_post_id, {})
        id2text_codetopk[this_post_id]["text"] =  []
        id2text_codetopk.setdefault(this_post_id, {})
        id2text_codetopk[this_post_id]["code"] =  {b_sorted_idx[0]: []}

        wordid_to_score_text = {}
        wordid_to_score_code = {}
        if highlight_mode == "textonly":
            #id2text_codetopk[this_post_id]["text"] = [x for x in range(this_cve_desc_end_idx, len(b_text_ids)) if b_text_ids[x] != 0]
            #if this_post_id == "CVE-2013-2130/f70f1086fd0c15d7fdb9eeef95dcefe9781ac3ab":
            #    import pdb; pdb.set_trace()
            for x in range(len(b_text_ids)):
                this_original_token = replace_tokens([b_text_tokens[x]])[0]
                if b_text_att[x] != 0 and re.search("[a-zA-Z0-9]", this_original_token):
                    wordid_to_score_text[x] = (b_text_ids[x], b_text_att[x], this_original_token)
            sorted_wordid_to_score_text = sorted(wordid_to_score_text.items(), key = lambda x:x[1][1], reverse=True)
            if len(sorted_wordid_to_score_text) > 0:
                #sorted_wordid_to_score = sorted_wordid_to_score_text[:topk]
                all_tokens = set([])
                for (each_key, each_id_token) in sorted_wordid_to_score_text:
                    each_id = each_id_token[0]
                    each_token = each_id_token[2]
                    id2text_codetopk.setdefault(this_post_id, {"text": []})
                    if len(id2text_codetopk[this_post_id]["text"]) < topk and (each_token.lower() not in all_tokens):
                        id2text_codetopk[this_post_id]["text"].append(each_key)
                        all_tokens.add(each_token.lower())
            #if this_post_id == "CVE-2017-5601/98dcbbf0bf4854bf987557e55e55fff7abbf3ea9":
            #    import pdb; pdb.set_trace()
        if highlight_mode == "codeonly":
            # if this_post_id == "CVE-2013-1800/e3da1212a1f84a898ee3601336d1dbbf118fb5f6":
            #     import pdb; pdb.set_trace()
            #id2text_codetopk[this_post_id]["code"] =  {b_sorted_idx[0]: [x for x in range(this_cve_desc_end_idx, len(b_code_ids)) if b_code_ids[x] != 0]}
            for x in range(len(b_code_ids)):
               this_origin_token = replace_tokens([b_code_tokens[x]])[0]
               if b_code_att[x] != 0 and re.search("[a-zA-Z0-9]", this_origin_token):
                    wordid_to_score_code[x] = (b_code_ids[x], b_code_att[x], this_origin_token)
            sorted_wordid_to_score_code = sorted(wordid_to_score_code.items(), key = lambda x:x[1][1], reverse=True)
            if len(sorted_wordid_to_score_code) > 0:
               #sorted_wordid_to_score = sorted_wordid_to_score_code[:topk]
               all_tokens = set([])
               for (each_key, each_id_token) in sorted_wordid_to_score_code:
                    each_id = each_id_token[0]
                    each_token = each_id_token[2]
                    id2text_codetopk.setdefault(this_post_id, {"code": {b_sorted_idx[0]: []}})
                    this_codetopk_list = id2text_codetopk[this_post_id]["code"][b_sorted_idx[0]]
                    if len(this_codetopk_list) < topk and (each_token.lower() not in all_tokens):
                        this_codetopk_list.append(each_key)
                        all_tokens.add(each_token.lower())
    
    modelClass = modelPred(params, test_data)

    ### code for reproducing the comprehensiveness and sufficiency in spreadsheet
    compute_faithfulness_tfidf(option, fold, filter_mode, params, suffix= "_" + option + "_" + highlight_mode, param_2_path=param_2_path, test_data_list=test_data_list, test_post_ids=test_post_ids, cve_end_idx_list=cve_end_idx_list, modelClass=modelClass, id2text_codetopk=id2text_codetopk, topk=topk)

