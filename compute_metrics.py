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

    lines = open(os.path.join(param_2_path, f"explanation_with_tfidf_{highlight_mode}_{fold}.json"), "r").readlines()
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
    #params["beta"] = 0.8 if path_idx == 0 else 0.7
    device = params["device"]
    model_name_hyphen = params["model_name"].replace("/", "-")

    param_2_path = params_2_model_save_path(params)

    param_2_path = 'explanations_dicts/' + param_2_path #.replace('Saved/', 'explanations_dicts/')
    if not os.path.exists('explanations_dicts'):
        os.makedirs('explanations_dicts')
    if not os.path.exists(param_2_path):
        os.makedirs(param_2_path)

    #for topk in [3, 5, 10, 20, 30]:
    #for highlight_mode in ["textonly", "codeonly"]:
    #    print(params["model_name"], fold, highlight_mode)
    print(debug(test_dataloader, test_data_list, params, highlight_mode, fold = fold, topk = topk, test_post_ids=test_post_ids))

