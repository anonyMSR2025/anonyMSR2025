import pandas as pd
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from functools import partial
import re
import bisect
import numpy as np
import sys, json, os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
from trace_params import set_params
import math

from Models.utils import return_params

from Preprocess.dataCollect import set_name
from Preprocess.process_ranktoken_attentions import tokenize_with_gt_attention_mask
from transformers import RobertaTokenizerFast,RobertaTokenizer
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
import argparse
from tqdm import tqdm
from Preprocess.rank_tokens import rank_tokens,split_by_camel_snake_and_lower
from Preprocess.rank_token_preprocess import load_cve_2_porject_info
tqdm.pandas()
import random
import swifter
from datasets import Features, Sequence, Value
from torch.utils.data import DataLoader

commit_list = [] #"7b4819124f49e205505785df0c7bf1362c61477b"]

def load_dataset_as_df(data_path):
    with open(data_path, 'r') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    print(f'length of data: {len(data)}')
    print(f'count of positive samples: {sum([d["label"] for d in data])}')
    data_df = pd.DataFrame(data)
    print(f'columens: {data_df.columns}')   
    return data_df

# def get_github_readme_title_tokens():
#     pass

def process_token(each_token):
    return re.sub(r'[^a-z]', '', re.sub(r'[^\x00-\x7F]+', '', each_token.lower().replace("_", "")))

def separate_diffs_attention(diff_list: list, attention_tokens:dict, commit:str):
    diff_list_len = [len(x) for x in diff_list]
    attention_dict_list = []
    for x in range(len(diff_list)):
        attention_dict_list.append({})
    cumsum_list = list(np.cumsum(diff_list_len))
    for x in range(len(cumsum_list)):
        cumsum_list[x] += x + 1 
    diff_list_len_cumsum = [0] + cumsum_list

    for each_word, each_tuple_list_and_score in attention_tokens.items():
        each_tuple_list = each_tuple_list_and_score[0]
        each_word_tfidf = each_tuple_list_and_score[1]
        for each_tuple in each_tuple_list:
            left_bin_idx = bisect.bisect_right(diff_list_len_cumsum, each_tuple[0]) - 1
            right_bin_idx = bisect.bisect_right(diff_list_len_cumsum, each_tuple[1]) - 1
            if left_bin_idx >= len(diff_list_len_cumsum) - 1: continue
            assert left_bin_idx == right_bin_idx or left_bin_idx == right_bin_idx - 1
            if left_bin_idx == right_bin_idx - 1:
                assert each_tuple[1] in diff_list_len_cumsum
            offset_left, offset_right = each_tuple[0] - diff_list_len_cumsum[left_bin_idx], each_tuple[1] - diff_list_len_cumsum[left_bin_idx] # offset needs to deduct left_bin_idx because previusly the character index (each_tuple) was computed in " ".join(diff_list), see rank_tokens.py:rank_tokens)
            if commit not in commit_list:
               if process_token(diff_list[left_bin_idx][offset_left:offset_right]) != process_token(each_word):
                   import pdb; pdb.set_trace()
               assert process_token(diff_list[left_bin_idx][offset_left:offset_right]) == process_token(each_word)
            attention_dict_list[left_bin_idx].setdefault(each_word, ([], each_word_tfidf))
            attention_dict_list[left_bin_idx][each_word][0].append([offset_left, offset_right])

    return attention_dict_list

def get_tokenizer(params):
    if "codebert" in params["model_name"] or "unixcoder" in params["model_name"]:
       tokenizer = RobertaTokenizerFast.from_pretrained(params["model_name"])
    return tokenizer 

def split_list_to_smaller_list(list_of_list, token_list_of_list, chunk_size, mode, params):
    token_lst = [each_list[x] for each_list in token_list_of_list for x in range(1, len(each_list) - 1)]
    token_map_l = [[fidx, x] for fidx, each_list in enumerate(token_list_of_list) for x in range(1, len(each_list) - 1)]
    # [file index, token index]
    lst = [each_list[x] for each_list in list_of_list for x in range(1, len(each_list) - 1)]

    assert len(lst) == len(token_lst)
    if params["model_name"] == "microsoft/codebert-base":
      lst = [lst[x] for x in range(len(lst)) if token_lst[x] not in [1437]]
      token_map_l = [token_map_l[x] for x in range(len(token_map_l)) if token_lst[x] not in [1437]]
    assert len(lst) == len(token_map_l)
    
    if mode == "token":
        return [[0] + lst[i:i + chunk_size] + [2] for i in range(0, len(lst), chunk_size)], [[[-1, -1]] + token_map_l[i:i + chunk_size] + [[-1, -1]] for i in range(0, len(token_map_l), chunk_size)]
    elif mode == "att":
        return [[0] + lst[i:i + chunk_size] + [0] for i in range(0, len(lst), chunk_size)], None

def map_token_id(token_id, params):
    if params["model_name"] == "microsoft/codebert-base" and token_id == 50265:
        return 50118
    else:
        return token_id

def tokenize_text(examples, tokenizer, params):
    '''
    use dataset map
    '''
    tokenized_inputs = {
        'cve_desc_ids': [],
        'cve_desc_attention': [],
        'commit_msg_ids': [],
        'commit_msg_attention': [],
        'diff_ids': [],
        'diff_attention': [],
        #'diff_': [],
        'label': [],
        'cve': [],
        'commit': [],
        #'diff_map_l': []
    }

    chunk_size = 64
   
    for cve_description_, commit_msg_, diff_l_, cve_attention_, commit_attention_, diff_attention_, label_, cve_, commit_ in zip(examples['cve_desc'], examples['commit_msg'], examples['diff'], examples['cve_attention'], examples['commit_attention'], examples['diff_code_attention'], examples['label'], examples['cve'], examples['commit']):
        # if commit_ != "343c7bd381b63e042d437111718918f951d9b30d":
        #     continue

        # cve_description = cve_description.lower()
        # commit_msg = commit_msg.lower()
        # diff_l = [x.lower() for x in diff_l]

        diff_attention_dict_list = separate_diffs_attention(diff_l_, json.loads(diff_attention_), commit_)

        tokenized_diff_list = [tokenize_with_gt_attention_mask(text = diff_l_[x], tokenizer = tokenizer, attention_tokens = diff_attention_dict_list[x], params=params, is_code=True, commit_id=commit_, part = "diff") for x in range(len(diff_l_))]

        token_list_of_list = [tokenized_diff["input_ids"] for tokenized_diff in tokenized_diff_list]
        att_list_of_list = [tokenized_diff['gt_attention'] for tokenized_diff in tokenized_diff_list]
        
        diff_input_ids, diff_inputids_map_l = split_list_to_smaller_list(token_list_of_list, token_list_of_list, chunk_size, mode = "token", params=params)
        diff_attention, _ = split_list_to_smaller_list(att_list_of_list, token_list_of_list, chunk_size, mode = "att", params=params)
        # import pdb; pdb.set_trace()  # 在这一步之前就已经被去掉了 1437
        assert len(diff_input_ids) == len(diff_attention)

        tokenized_cve_desc = tokenize_with_gt_attention_mask(text = cve_description_, tokenizer = tokenizer, attention_tokens = json.loads(cve_attention_), params=params, is_code=False, commit_id=commit_, part = "cve")
        cve_attention = tokenized_cve_desc['gt_attention']
        assert len(tokenized_cve_desc['input_ids']) == len(cve_attention)

        tokenized_commit_msg = tokenize_with_gt_attention_mask(text = commit_msg_, tokenizer = tokenizer, attention_tokens = json.loads(commit_attention_), params=params, is_code=False, commit_id=commit_, part = "commit")
        commit_msg_attention = tokenized_commit_msg['gt_attention']
        assert len(tokenized_commit_msg['input_ids']) == len(commit_msg_attention)
        
        tokenized_inputs['cve_desc_ids'].append([map_token_id(x, params) for x in tokenized_cve_desc['input_ids']])
        tokenized_inputs['cve_desc_attention'].append(cve_attention)
        tokenized_inputs['commit_msg_ids'].append([map_token_id(x, params) for x in tokenized_commit_msg['input_ids']])
        tokenized_inputs['commit_msg_attention'].append(commit_msg_attention)
        tokenized_inputs['diff_ids'].append([[map_token_id(x, params) for x in each_ids] for each_ids in diff_input_ids])
        tokenized_inputs['diff_attention'].append(diff_attention)
        tokenized_inputs['label'].append(int(label_))
        tokenized_inputs['cve'].append(cve_)
        tokenized_inputs['commit'].append(commit_)
        #tokenized_inputs['diff_map_l'].append(diff_inputids_map_l)

        # if commit_ == "343c7bd381b63e042d437111718918f951d9b30d":
        #     import pdb; pdb.set_trace()
                        
    return tokenized_inputs

def step4_rank_code_by_tfidf(data_dir, params):

    is_rank_contain_attention_first = True

    tokenizer = RobertaTokenizer.from_pretrained(params["model_name"], do_lower_case=False)
    id_to_token_map = {v: k for k, v in tokenizer.get_vocab().items()}
    title_repo = json.load(open(os.path.join(data_dir, "cve_2_porject_info.json"), "r"))

    token2alldf = json.load(open(os.path.join(data_dir, "token2df.json"), "r"))
    cve2cvedf = json.load(open(os.path.join(data_dir, "cve2cvedf.json"), "r"))
    cve2count = json.load(open(os.path.join(data_dir, "cve2count.json"), "r"))

    for fold in ["train", "test", "valid"]:
        output_data_dir = os.path.join(data_dir, set_name(params))
        data_fold = json.load(open(os.path.join(output_data_dir, f'output_Dataset_{fold}_2.json'), "r"))
        new_data_fold = {}

        for key, val in tqdm(data_fold.items()):
            cve = key.split("/")[0]
            if len(val["diff_ids"]) == 0: continue

            stop_tokens = title_repo[cve]["title"] + title_repo[cve]["owner_repo"].split("\/")
            cve_desc = "".join([id_to_token_map[x].replace("Ġ", " ") if x != 50118 else "\n" for x in val["cve_desc_ids"]]).replace("<s>", "").replace("</s>", "")
            cve_tokens = split_by_camel_snake_and_lower(cve_desc, stop_tokens).keys()
            cve_tfidf = compute_document_tfidf(cve_tokens, cve2cvedf.get(cve, {}), token2alldf, cve2count[cve] + 10, 3500)

            code_ids = val["diff_ids"]
            
            idx2score = {}
            for idx in range(len(code_ids)):
                each_diff_code = code_ids[idx]
                
                code = "".join([id_to_token_map[x].replace("Ġ", " ") if x != 50118 else "\n" for x in each_diff_code]).replace("<s>", "").replace("</s>", "")
                code_tokens = split_by_camel_snake_and_lower(code, stop_tokens).keys()
                code_tfidf = compute_document_tfidf(code_tokens, cve2cvedf.get(cve, {}), token2alldf, cve2count[cve] + 10, 3500)

                tfidf_score = compute_cosine_similarity(cve_tfidf, code_tfidf)
                idx2score[idx] = tfidf_score

            sorted_idx = sorted(idx2score, key=lambda x: idx2score[x], reverse=True)

            if is_rank_contain_attention_first: # if ranking the snippets with attention tokens before those without
                sorted_idx_w_att = [x for x in sorted_idx if sum(val["diff_attention"][x]) > 0]
                sorted_idx_wo_att = [x for x in sorted_idx if sum(val["diff_attention"][x]) == 0]

                sorted_idx = sorted_idx_w_att + sorted_idx_wo_att
            val["sorted_idx"] = sorted_idx
            new_data_fold[key] = val
        json.dump(new_data_fold, open(os.path.join(output_data_dir, f'output_Dataset_{fold}_ranked_attfirst.json'), "w"))


def compute_cosine_similarity(query_dict,
                              candidate_dict):
    score = 0
    intersection = set(query_dict.keys()) & set(candidate_dict.keys())

    prod, sqr1, sqr2 = 0, 0, 0
    for token in intersection:
        prod += query_dict[token] * candidate_dict[token]

    for cnt in query_dict.values(): sqr1 += cnt ** 2
    for cnt in candidate_dict.values(): sqr2 += cnt ** 2

    if math.isclose(sqr1, 0) or math.isclose(sqr2, 0): score = 0
    else: score = prod / math.sqrt(sqr1) / math.sqrt(sqr2)
    return score

                
def compute_document_tfidf(tokens_list,
                           token2df, token2alldf, samecve_count, all_count):

    tokens_dict = dict()
    for token in tokens_list:
        tokens_dict.setdefault(token, 0)
        tokens_dict[token] += 1

    for word, tf in tokens_dict.items():
        offset = 0.01
        df1 = token2df.get(word, 0) + offset
        df2 = token2alldf.get(word, 0) + offset
        samecve_idf = math.log( samecve_count / df1, 2)
        all_idf = math.log(all_count / df2, 2)
        tokens_dict[word] = tf * all_idf * samecve_idf

    return tokens_dict         

def step3_tokenize(data_dir, params):  # use output of step1 and step2
    num_proc = None #40
    batch_size =  16
    VALID_DATA_PATH = os.path.join(data_dir, 'tf_idf_filtered_bert_input_valid.json')
    DOWNSAMPLED_TRAIN_DATA_PATH = os.path.join(data_dir, 'tf_idf_filtered_bert_input_train_downsampled.json')
    TEST_DATA_PATH = os.path.join(data_dir, 'tf_idf_filtered_bert_input_test.json')
    data_path = {
        'VALID_DATA_PATH': VALID_DATA_PATH,
        'DOWNSAMPLED_TRAIN_DATA_PATH': DOWNSAMPLED_TRAIN_DATA_PATH,
        'TEST_DATA_PATH': TEST_DATA_PATH
    }
    max_diff_code_size = 20
    
    for fold in ["train", "valid", "test"]: #, "test", "valid"]:
       print("processing", fold)
       output_Dataset = dict() # key: cve_id+commit_id, values: data and split
  
       if fold == "train":
           suffix1 = "train_downsampled"
           suffix2 = "DOWNSAMPLED_TRAIN"
       else:
           suffix1 = fold
           suffix2 = fold.upper()

       attention_sep = json.load(open(os.path.join(data_dir, f"{suffix1}_sep_2.json"), "r"))
       cve_sep_list, commit_msg_list, diff_sep_list = attention_sep["cve_sep"], attention_sep["commit_msg_sep"], attention_sep["diff_sep"]

       cve_sep_list = [json.dumps(x) for x in cve_sep_list]
       commit_msg_list = [json.dumps(x) for x in commit_msg_list]
       diff_sep_str_list = [json.dumps(x) for x in diff_sep_list]

       this_data_set = load_dataset('json', data_files=data_path[f'{suffix2}_DATA_PATH'])["train"]

       this_data_df = this_data_set.to_pandas()
       import pdb; pdb.set_trace()
       # remove examples where there are too many files in diff code
       diff_list = []
       for x in range(len(this_data_df)):
           diff_list.append(this_data_df["diff"].iloc[x][:max_diff_code_size])
       this_data_df["diff"] = diff_list
       if "cve_attention_tokens" in this_data_df.columns:
            this_data_df.drop(columns=["cve_attention_tokens"], inplace=True)
       this_data_df["cve_attention"] = cve_sep_list
       this_data_df["commit_attention"] = commit_msg_list
       this_data_df["diff_code_attention"] = diff_sep_str_list

       #commit_id_subset = [x for x in range(len(this_data_df)) if this_data_df["commit"].iloc[x] == "e46ca3ac46af839208b9cf9206e2af99dfc0e3b5"] # "666ed7facf4524bf6d19b11b20faa2cf93fdf591" "e46ca3ac46af839208b9cf9206e2af99dfc0e3b5"]
       #this_data_set = Dataset.from_dict(Dataset.from_pandas(this_data_df)[:100])
       this_data_set = Dataset.from_pandas(this_data_df)
    #    import pdb; pdb.set_trace()  # sum([sum([int(x == 1437) for x in each_ids]) for each_ids in diff_input_ids])
       this_data_set = this_data_set.map(partial(tokenize_text, tokenizer=get_tokenizer(params), params=params), batched=True, batch_size=batch_size, num_proc=num_proc)
       this_data_df = this_data_set.to_pandas()
       this_data_df.drop(columns=["cve_desc", 'commit_msg', 'diff', 'cve_attention', 'commit_attention', 'diff_code_attention'], inplace=True)
       this_data_set = Dataset.from_pandas(this_data_df)

       dataiter = DataLoader(this_data_set, batch_size=20, collate_fn=lambda x: x)   # do nothing for collate_fn
       for batch in tqdm(dataiter):
           for one_data in batch:
               output_Dataset[one_data["cve"] + '/' + one_data['commit']] = one_data
       print('train data tokenized')
       output_data_dir = os.path.join(data_dir, set_name(params))
       if not os.path.exists(output_data_dir):
           os.makedirs(output_data_dir)
       with open(os.path.join(output_data_dir, f'output_Dataset_{fold}_2.json'), 'w') as f:
           json.dump(output_Dataset, f)
        
def load_cve_2_attention_tokens(threshold: float = 0.0, file_path :str = './cve_output_attention_tokens.xlsx'):
    '''
    '''
    data_df = pd.read_excel(file_path, engine='openpyxl')
    output_cve_2_attention_tokens = {}
    for idx, row in data_df.iterrows():
        # print(row)
        cve = row['cve']
        attention_tokens_raw = row['cve_output_tokens']
        commit = row['commit']
        commit_id = commit.split('commit/')[-1]
        attention_tokens = [item[0] for item in eval(attention_tokens_raw) if item[1] > threshold]
        output_cve_2_attention_tokens[(cve, commit_id)] = attention_tokens
    return output_cve_2_attention_tokens

def load_data_savememory(TOTAL_DATA_PATH):
    '''
    only load metadata
    '''
    data = []
    total_cve = []

    with open(TOTAL_DATA_PATH, 'r') as f:
        for line in tqdm(f, desc='load metadata'):
            cur_cve_data = json.loads(line)

            for index, (commit, label) in enumerate(zip(cur_cve_data['filtered_candidate_commits'], cur_cve_data['filtered_candidate_commit_labels']), start=1):
                # 将每条记录直接添加到 data
                data.append({
                    # 'cve_desc': cur_cve_data['cve_description'],
                    # 'commit_msg': cur_cve_data['commit2str'][commit]['commit_msg'],
                    # 'diff': cur_cve_data['commit2str'][commit]['diffs'],
                    'label': label,
                    'cve': cur_cve_data['cve'],
                    'rank': index,
                    'commit': commit,
                })

            # 将当前 CVE 添加到 total_cve
            total_cve.append(cur_cve_data['cve'])

            # 清理当前数据以释放内存
            cur_cve_data.clear()

            # 如果内存使用过多，可以在这里定期释放内存
            if len(data) > 50000:  # 设置一个合理的阈值（如 50000 条记录）
                pass  # 不需要任何特殊处理，因为 data 不会清理（除非逻辑改变）

    return data, total_cve


def load_data(TOTAL_DATA_PATH):
    data = []
    total_cve = []
    #cve2attention_tokens = load_cve_2_attention_tokens(file_path='./cve_output_attention_tokens.xlsx')
    with open(TOTAL_DATA_PATH, 'r') as f:
        for line in tqdm(f):  
            cur_cve_data = json.loads(line)
            # print(cur_cve_data.keys())
            # raise Exception('stop here')
            # have maken sure that the commits are ordered by the bm25 score 
            for index, (commit, label) in enumerate(zip(cur_cve_data['filtered_candidate_commits'], cur_cve_data['filtered_candidate_commit_labels']), start=1):
                data.append({
                    'cve_desc': cur_cve_data['cve_description'],
                    'commit_msg': cur_cve_data['commit2str'][commit]['commit_msg'],
                    'diff': [diff for diff in cur_cve_data['commit2str'][commit]['diffs']],
                    'label': label,
                    'cve': cur_cve_data['cve'],
                    'rank': index,
                    'commit': commit,
                   # 'cve_attention_tokens': cve2attention_tokens.get((cur_cve_data['cve'], commit), [])
                })
                # diff_token_count.append(len(split_text(cur_cve_data['commit2str'][commit]['cve_description'] + ' [SEP] ' + cur_cve_data['commit2str'][commit]['diffs'])))
                # vulfiles_token_count.append(len(split_text(cur_cve_data['commit2str'][commit]['cve_description'] + ' [SEP] ' + cur_cve_data['commit2str'][commit]['vulnerable_files'])))
            total_cve.append(cur_cve_data['cve'])
            cur_cve_data.clear()
    return data, total_cve

def split_data(data, test_size=0.3, train_thread_shold=120, test_thread_shold=130):
    '''
    customized split function to ensure all commits of a cve are in the same set
        and ensure the propotion of test data is around given test_size
    '''
    test_ratio = 1
    total_cve = list(set([item['cve'] for item in data]))
    while abs(test_ratio - test_size) > 0.05:
        random.shuffle(total_cve)
        test_cve = total_cve[:int(len(total_cve) * test_size)]
        train_cve = total_cve[int(len(total_cve) * test_size):]
        train_data = [d for d in data if d['cve'] in train_cve and d['rank'] <= train_thread_shold]
        test_data = [d for d in data if d['cve'] in test_cve and d['rank'] <= test_thread_shold]
        test_ratio = len(test_data) / len(data)
    return train_data, test_data

def split_diff_by_file(diff_content):
    if isinstance(diff_content, list):
        return diff_content
    file_diff_pattern = r"(?=diff --git a/.*)"
    file_diffs = re.split(file_diff_pattern, diff_content)
    file_diffs = [diff.strip() for diff in file_diffs if diff.strip()]
    
    # import pdb; pdb.set_trace()
    return file_diffs

def apply_partition(TOTAL_DATA_PATH, TRAIN_DATA_PATH, VALID_DATA_PATH, TEST_DATA_PATH, DOWNSAMPLED_TRAIN_DATA_PATH, partition_d):
    train_data_output_f = open(TRAIN_DATA_PATH, 'w')
    valid_data_output_f = open(VALID_DATA_PATH, 'w')
    test_data_output_f = open(TEST_DATA_PATH, 'w')
    downsampled_train_data_output_f = open(DOWNSAMPLED_TRAIN_DATA_PATH, 'w')

    skipped_commit_count = 0

    with open(TOTAL_DATA_PATH, 'r') as f:
        for line in tqdm(f, desc='apply partition'):  
            cur_cve_data = json.loads(line)
            # have maken sure that the commits are ordered by the bm25 score
            owner = cur_cve_data['owner']
            repo = cur_cve_data['repo']
            for index, (commit, label) in enumerate(zip(cur_cve_data['filtered_candidate_commits'], cur_cve_data['filtered_candidate_commit_labels']), start=1):
                cur_data = {
                    'cve_desc': cur_cve_data['cve_description'],
                    'commit_msg': cur_cve_data['commit2str'][commit]['commit_msg'],
                    'diff': split_diff_by_file(cur_cve_data['commit2str'][commit]['diffs']),
                    'owner_repo': f"{owner}/{repo}",
                    'label': label,
                    'cve': cur_cve_data['cve'],
                    'rank': index,
                    'commit': commit,
                }
                partition_k = f"{cur_cve_data['cve']}/{commit}"
                if partition_k not in partition_d:
                    skipped_commit_count += 1
                    continue

                if partition_d[partition_k] == 'train':
                    train_data_output_f.write(json.dumps(cur_data) + '\n')
                elif partition_d[partition_k] == 'valid':
                    valid_data_output_f.write(json.dumps(cur_data) + '\n')
                elif partition_d[partition_k] == 'test':
                    test_data_output_f.write(json.dumps(cur_data) + '\n')
                elif partition_d[partition_k] == 'train_downsample':
                    output_str = json.dumps(cur_data)
                    train_data_output_f.write(output_str + '\n')
                    downsampled_train_data_output_f.write(output_str + '\n')
                # else:
                    # import pdb; pdb.set_trace()

    print(f'skipped commit count: {skipped_commit_count}')


def prepareData(data_dir, split: bool = False, random_state: int = 42, balance_ratio=30):
    TOTAL_DATA_PATH = os.path.join(data_dir, 'tf_idf_filtered_bert_input.json')
    TRAIN_DATA_PATH = os.path.join(data_dir, 'tf_idf_filtered_bert_input_train.json')
    VALID_DATA_PATH = os.path.join(data_dir, 'tf_idf_filtered_bert_input_valid.json')
    DOWNSAMPLED_TRAIN_DATA_PATH = os.path.join(data_dir, 'tf_idf_filtered_bert_input_train_downsampled.json')
    TEST_DATA_PATH = os.path.join(data_dir, 'tf_idf_filtered_bert_input_test.json')

    data_path = {
        'VALID_DATA_PATH': VALID_DATA_PATH,
        'DOWNSAMPLED_TRAIN_DATA_PATH': DOWNSAMPLED_TRAIN_DATA_PATH,
        'TEST_DATA_PATH': TEST_DATA_PATH
    }
    exist_flag = True
    if split:
        # Split the data into train and test
        data, total_cve = load_data_savememory(TOTAL_DATA_PATH)
        print(f'length of data: {len(data)}')
        print(f'count of positive samples: {sum([d["label"] for d in data])}')
        print(f'positive rate: {sum([d["label"] for d in data]) / len(data)}')
        # raise Exception('stop here')

        # train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
        train_valid_data, test_data = split_data(data)
        train_data, valid_data = split_data(train_valid_data)
        # import pdb; pdb.set_trace()
        # save partition
        partition_d = dict()
        for item in train_data:
            partition_d[f"{item['cve']}/{item['commit']}"] = 'train'
        for item in valid_data:
            partition_d[f"{item['cve']}/{item['commit']}"] = 'valid'
        for item in test_data:
            partition_d[f"{item['cve']}/{item['commit']}"] = 'test'
        

        # down sample train dataset
        negative_samples = [sample for sample in train_data if sample['label'] == 0]
        positive_samples = [sample for sample in train_data if sample['label'] == 1]
        downsampled_negative_samples = random.sample(negative_samples, int(len(positive_samples)) * balance_ratio) # balance the dataset, negative samples are [balance_ratio] times of positive samples
        downsampled_train_dataset = positive_samples + downsampled_negative_samples
        for item in downsampled_train_dataset:
            partition_d[f"{item['cve']}/{item['commit']}"] = 'train_downsample'
        with open('./partition_d.json', 'w') as f:
            json.dump(partition_d, f)
        # import pdb; pdb.set_trace()

        apply_partition(TOTAL_DATA_PATH, TRAIN_DATA_PATH, VALID_DATA_PATH, TEST_DATA_PATH, DOWNSAMPLED_TRAIN_DATA_PATH, partition_d)
        return data_path

    # no need to return dataset object
    # downsampled_train_dataset = load_dataset('json', data_files={"train": DOWNSAMPLED_TRAIN_DATA_PATH}, split='train')
    # # train_dataset = load_dataset('json', data_files={"train": TRAIN_DATA_PATH, "test": TEST_DATA_PATH}, split='train')
    # valid_dataset = load_dataset('json', data_files={"valid": VALID_DATA_PATH}, split='valid')
    # test_dataset = load_dataset('json', data_files={"test": TEST_DATA_PATH}, split='test')
    
    # # return downsampled_train_dataset.from_dict(downsampled_train_dataset[:1000]), Dataset.from_dict(test_dataset[:1000])
    # return downsampled_train_dataset, valid_dataset, test_dataset, data_path

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)

    parser.add_argument('--path',
                           type=str,
                           help='The path to json containining the parameters',
                           default="./best_model_json/trace_params.json")

    args = parser.parse_args()
    with open(args.path, mode='r') as f:
        params = json.load(f)
    params["model_name"] = args.model_name
        
    
    # step 1: extract train/test/valid json file
    if args.step == 1:
        prepareData(data_dir=args.data_dir, split=True)

    if args.step == 2:
        load_cve_2_porject_info(data_dir=args.data_dir)
    # step 2: use tf-idf to compute cve_output_attention.xlsx
    if args.step == 3:
        rank_tokens(data_dir=args.data_dir)
    
    # step 3: tokenize and merge tf-idf ranked tokens
    if args.step == 4:
       step3_tokenize(data_dir = args.data_dir, params=params)

    if args.step == 5:
       step4_rank_code_by_tfidf(data_dir = args.data_dir, params=params)
    
    # train_data = collect_data(params)

    # python process_data.py --data_dir ../Data/ --path ../best_model_json/trace_params.json --model_name microsoft/codebert-base
    # python process_data.py --data_dir ../../timeFilter/output_data/ --model_name microsoft/codebert-base --path ../best_model_json/trace_params.json --step 1
    
