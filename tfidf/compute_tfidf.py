#def is_valid(group_df):
import tqdm
from collections import Counter, defaultdict
from multiprocessing import Pool
import re
from datetime import datetime, timezone, timedelta
import json
from typing import List, Tuple, Dict
from tfidf import cosine_similarity_inverted_indenx
import numpy as np
import pandas as pd
import os
import math
from utils.debug import *

from compute_df import parse_datetime, split_by_camel_snake_and_lower

from utils.repo_utils import dict_update, read_commit_file, parse_commit_file
from functools import partial

if not os.path.exists("data/tfidf"):
    os.makedirs("data/tfidf")


global commit_idf_dict, diff_idf_dict
with open('./tfidf_data/idf_commit.json', 'r') as fp:
    commit_idf_dict = json.load(fp) # ['idf']
print(len(commit_idf_dict['idf']))
small_commit = [item for item, v in commit_idf_dict['idf'].items() if v < 0]

with open('./tfidf_data/idf_diff.json', 'r') as fp:
    diff_idf_dict = json.load(fp) # 'idf']
print(len(diff_idf_dict['idf']))
small_diff = [(item, v) for item, v in diff_idf_dict['idf'].items() if v < 0]

# import pdb; pdb.set_trace()  # DEBUG


def compute_document_tfidf(document_word_cnt_dict: dict,
                           idf_vocab: dict,
                           is_query: bool) -> Dict[str, float]:
    '''
    tfidf -> bm25
    Score(Q, d) = \sum_{i=1}^{n} W_i * R(q_i, d)

    W_i = \frac{N - n_i + 0.5}{n_i + 0.5}, compute in function compute_idf
    R(q_i, d) is computed in this function, but separate into two parts
    R(q_i, d) = \frac{qf_i(k_2 + 1)}{qf_i + k_2} * \frac{tf_i(k_1 + 1)}{tf_i + K}
    1. for query, compute \frac{qf_i(k_2 + 1)}{qf_i + k_2} 
    2. for document, compute W_i * \frac{tf_i(k_1 + 1)}{tf_i + K}

    compute the inner product of two vectors can get the final Score(Q, d)
    NOTE: the vector should not be normalized, not cosine similarity
    '''
    document_word_tfidf_dict = dict()
    #############################################START HERE#############################################
    for word, tf in document_word_cnt_dict.items():
        if word not in idf_vocab['idf']: continue
        idf = idf_vocab['idf'][word] 
        if idf > 0:
            document_word_tfidf_dict[word] = idf * tf
    ##############################################END HERE#############################################
    return document_word_tfidf_dict


def compute_document_bm25(document_word_cnt_dict: dict,
                           idf_vocab: dict,
                           is_query: bool) -> Dict[str, float]:
    '''
    tfidf -> bm25
    Score(Q, d) = \sum_{i=1}^{n} W_i * R(q_i, d)

    W_i = \frac{N - n_i + 0.5}{n_i + 0.5}, compute in function compute_idf
    R(q_i, d) is computed in this function, but separate into two parts
    R(q_i, d) = \frac{qf_i(k_2 + 1)}{qf_i + k_2} * \frac{tf_i(k_1 + 1)}{tf_i + K}
    1. for query, compute \frac{qf_i(k_2 + 1)}{qf_i + k_2} 
    2. for document, compute W_i * \frac{tf_i(k_1 + 1)}{tf_i + K}

    compute the inner product of two vectors can get the final Score(Q, d)
    NOTE: the vector should not be normalized, not cosine similarity
    '''
    k_1, k_2, b = 1.2, 1, 0.75

    document_word_tfidf_dict = dict()
    #############################################START HERE#############################################
    # Question 5 (15 pts)
    D = 0  # document lenght
    for word, tf in document_word_cnt_dict.items():
        if word not in idf_vocab['idf']: continue
        D += tf
    avgdl = idf_vocab['avg_doc_len']  # average document length
    # import pdb; pdb.set_trace()
    for word, tf in document_word_cnt_dict.items():
        if word not in idf_vocab['idf']: continue
        # if word == 'url':
            # import pdb; pdb.set_trace()  # DEBUG
        if is_query:
            # \frac{qf_i(k_2 + 1)}{qf_i + k_2}
            document_word_tfidf_dict[word] = (tf * (k_2 + 1)) / (tf + k_2)

        else:
            # W_i * \frac{tf_i(k_1 + 1)}{tf_i + K}
            # K = k_1 * (1 - b + b * \frac{D}{avgdl})
            # W_i = \frac{N - n_i + 0.5}{n_i + 0.5}
            K = k_1 * (1 - b + b * D / avgdl)
            idf = idf_vocab['idf'][word] 
            if idf < 0:
                idf = 0
            document_word_tfidf_dict[word] = idf * (tf * (k_1 + 1)) / (tf + K)
    ##############################################END HERE##############################################
    
    # import pdb; pdb.set_trace()  # DEBUG
    return document_word_tfidf_dict

def read_local_idf(owner, repo):
    try:
        with open(f"tfidf_data/inrepo_idf/idf_{owner}@@{repo}_commit.json", "r") as f:
            idf_commit = json.load(f)
        with open(f"tfidf_data/inrepo_idf/idf_{owner}@@{repo}_diff.json", "r") as f:
            idf_diff = json.load(f)
    except:
        return None, None
    return idf_commit, idf_diff

def get_tfidf(each_item):
    global linecount
    owner_repo, startends = each_item
    owner, repo = owner_repo.split("@@")

    cves = [x[0] for x in startends]
    cve_2_desc = {x[0]: x[1] for x in startends}
    cve_2_refs = {x[0]: x[2] for x in startends}
    cve_startends = [x[3] for x in startends]
    cve_2_startends = {x[0]: x[3] for x in startends}

    # import pdb; pdb.set_trace()  # DEBUG cve_startends & cves
    print("begin", owner, repo)

    class tfidf_processer():
        def __init__(self):
            self.df_diff = Counter()
            self.df_commit = Counter()
            self.N = 0
        
        def tfidf_process_file(self, commit_content, diff_content, idf_commit, idf_diff):
            commit_tokens, commit_tokens_set = split_by_camel_snake_and_lower(commit_content, [])
            diff_tokens, diff_tokens_set = split_by_camel_snake_and_lower(diff_content, [])
            commit_token_2_tf = Counter(commit_tokens)
            diff_token_2_tf = Counter(diff_tokens)
            filtered_tf_commit, filtered_tf_diff = dict(), dict()
            commit_tfidf = dict()
            diff_tfidf = dict()
            for each_token in commit_tokens_set:
                if len(each_token) >= 20: continue
                filtered_tf_commit[each_token] = commit_token_2_tf[each_token]
                
            for each_token in diff_tokens_set:
                if len(each_token) >= 20: continue
                filtered_tf_diff[each_token] = diff_token_2_tf[each_token]

            commit_tfidf = compute_document_bm25(filtered_tf_commit, idf_commit, is_query=False)
            diff_tfidf = compute_document_bm25(filtered_tf_diff, idf_diff, is_query=False)

            return commit_tfidf, diff_tfidf

    processer = tfidf_processer()
    inputed_func = partial(processer.tfidf_process_file, idf_commit=commit_idf_dict, idf_diff=diff_idf_dict)


    
    if os.path.exists(f"data/tfidf/tfidf_{owner}@@{repo}_commit.json") and os.path.exists(f"data/tfidf/tfidf_{owner}@@{repo}_diff.json"):
        try:  # check completeness
            with open(f"data/tfidf/tfidf_{owner}@@{repo}_commit.json", 'r') as fp:
                test_r = json.load(fp)
            with open(f'data/tfidf/tfidf_{owner}@@{repo}_diff.json', 'r') as fp:
                test_r = json.load(fp)
            embed_exist_flag = True
        except json.decoder.JSONDecodeError as e:
            embed_exist_flag = False
        
    else:
        embed_exist_flag = False
        
    #part2tfidfdict_commit = {key: part2tfidfdict_commit[key] for key in part2tfidfdict_commit if part2tfidfdict_commit[key] > 1}
    #part2tfidfdict_diff = {key: part2tfidfdict_diff[key] for key in part2tfidfdict_diff if part2tfidfdict_diff[key] > 1}
    print(owner, repo)
    if not embed_exist_flag:
        part2tfidfdict_commit, part2tfidfdict_diff = dict(), dict()
        cve_2_candidates, commit_2_embed_res = read_commit_file(owner, repo, cve_2_startends, inputed_func)
        local_idf_commit, local_idf_diff = read_local_idf(owner, repo)  # get local idf
        if local_idf_commit is None or local_idf_diff is None:
            return []
        # import pdb; pdb.set_trace()  # DEBUG
        for commit_id, (cur_commit_tfidf_commit, cur_diff_tfidf_diff) in commit_2_embed_res.items():
            part2tfidfdict_commit[commit_id] = {token: tfidf * max(local_idf_commit[token] - 1, 0) for token, tfidf in cur_commit_tfidf_commit.items() if token in local_idf_commit}
            part2tfidfdict_diff[commit_id] = {token: tfidf * max(local_idf_diff[token] - 1, 0) for token, tfidf in cur_diff_tfidf_diff.items() if token in local_idf_diff}
        del commit_2_embed_res
        json.dump(part2tfidfdict_commit, open(f"data/tfidf/tfidf_{owner}@@{repo}_commit.json", "w"))
        json.dump(part2tfidfdict_diff, open(f"data/tfidf/tfidf_{owner}@@{repo}_diff.json", "w"))
    else:
        def fake_func(x1, x2):
            return None
        cve_2_candidates, _ = read_commit_file(owner, repo, cve_2_startends, fake_func)
        with open(f"data/tfidf/tfidf_{owner}@@{repo}_commit.json", "r") as f:
            part2tfidfdict_commit = json.load(f)
        # try:
        with open(f"data/tfidf/tfidf_{owner}@@{repo}_diff.json", "r") as f:
            part2tfidfdict_diff = json.load(f)
        # except json.decoder.JSONDecodeError:
        #     return None
            
    linecount += 1
    #return Counter(part2tfidfdict_commit), Counter(part2tfidfdict_diff)

    # # generate cve_2_embed
    cve_2_embed = dict()
    for cve, candidates in cve_2_candidates.items():
        cve_2_embed[cve] = {'commit': dict(), 'diff': dict()}
        cve_desc = cve_2_desc[cve]
        tokens, tokens_set = split_by_camel_snake_and_lower(cve_desc, [])
        token2tf = Counter(tokens)
        filter_cve_tf = dict()
        for each_token in tokens_set:
            if len(each_token) >= 20: continue
            filter_cve_tf[each_token] = token2tf[each_token]

        cve_2_embed[cve]['commit'] = compute_document_bm25(filter_cve_tf, commit_idf_dict, is_query=True)
        cve_2_embed[cve]['diff'] = compute_document_bm25(filter_cve_tf, diff_idf_dict, is_query=True)
        

    # compute tfidf recall
    # import pdb; pdb.set_trace()  # DEBUG
    output_res = []
    score_details = []
    for cve, candidates in cve_2_candidates.items():
        candidates = list(set(candidates))
        print(cve, len(candidates)) # DEBUG
        candidate_commit_2_idx = {commit: idx for idx, commit in enumerate(candidates)}
        # import pdb; pdb.set_trace()  # DEBUG
        cve_embed_commit, cve_embed_diff = cve_2_embed[cve]['commit'], cve_2_embed[cve]['diff']
        candidates_embed_commit = [part2tfidfdict_commit.get(x, dict()) for x in candidates]
        candidates_embed_diff = [part2tfidfdict_diff.get(x, dict()) for x in candidates]
        similarities_commit = cosine_similarity_inverted_indenx(cve_embed_commit, candidates_embed_commit)
        similarities_diff = cosine_similarity_inverted_indenx(cve_embed_diff, candidates_embed_diff)
        similarities = similarities_commit * 0.6 + similarities_diff * 0.4
        
        patches = cve_2_refs[cve]
        sorted_indices = list(np.argsort(similarities)[::-1])
        # output details
        # Export similarity scores and ranks for each commit
        if not os.path.exists(f"tfidf_data/tfidf_detail"):
            os.makedirs(f"tfidf_data/tfidf_detail")
        
        for idx, commit_id in enumerate(candidates):
            sim_score = similarities[idx]
            rank = sorted_indices.index(idx) + 1
            score_details.append(f"{cve}\t{commit_id}\t{sim_score:.6f}\t{rank}")
        
        

        for patch in patches:
            if patch not in candidate_commit_2_idx:
                output_res.append((cve, repo, owner, patch, -1))
            else:
                patch_idx = candidate_commit_2_idx[patch]
                rank = sorted_indices.index(patch_idx) + 1
                output_res.append((cve, repo, owner, patch, rank))
                # if rank > 100 or cve == 'CVE-2021-21321': # 导出 bad cases
                #     # Only print matching tokens between CVE and patch
                #     f_commit_msg = open(f'data/commits/commits_{owner}@@{repo}.txt', 'r', encoding='iso-8859-1')
                #     f_diff = open(f'data/diff/diff_{owner}@@{repo}.txt', 'r', encoding='iso-8859-1')
                #     cve_tokens = set(cve_2_embed[cve]['diff'].keys())

                #     commit_msg_infos = parse_commit_file(owner, repo, 'commit_msg', None) # DEBUG
                #     commit_2_commit_info = {item.commit_id: item for item in commit_msg_infos}
                #     diff_infos = parse_commit_file(owner, repo, 'diff', None) # DEBUG
                #     commit_2_diff_info = {item.commit_id: item for item in diff_infos}
                #     # import pdb; pdb.set_trace()  # DEBUG
                    
                #     patch_commit_matches = {k:v for k,v in part2tfidfdict_commit[patch].items() if k in cve_tokens}
                #     patch_diff_matches = {k:v for k,v in part2tfidfdict_diff[patch].items() if k in cve_tokens}
                #     print('cve:', cve_2_desc[cve])
                #     print('patch commit matching tokens:', patch_commit_matches) 
                #     print('patch diff matching tokens:', patch_diff_matches)
                #     patch_commit_msg_info = commit_2_commit_info[patch]
                #     patch_diff_info = commit_2_diff_info[patch]
                #     f_commit_msg.seek(patch_commit_msg_info.start_offset)
                #     f_diff.seek(patch_diff_info.start_offset)
                #     patch_commit_msg = f_commit_msg.read(patch_commit_msg_info.end_offset - patch_commit_msg_info.start_offset)
                #     patch_diff = f_diff.read(min(1000, patch_diff_info.end_offset - patch_diff_info.start_offset))
                #     print('patch commit msg:', patch_commit_msg)
                #     print('patch diff:', patch_diff)

                #     rank_1_commit_id = candidates[sorted_indices[0]]
                #     rank_1_commit_matches = {k:v for k,v in part2tfidfdict_commit[rank_1_commit_id].items() if k in cve_tokens}
                #     rank_1_diff_matches = {k:v for k,v in part2tfidfdict_diff[rank_1_commit_id].items() if k in cve_tokens}
                #     print('rank 1 commit matching tokens:', rank_1_commit_matches)
                #     print('rank 1 diff matching tokens:', rank_1_diff_matches)
                #     rank_1_commit_msg_info = commit_2_commit_info[rank_1_commit_id]
                #     rank_1_diff_info = commit_2_diff_info[rank_1_commit_id]
                #     f_commit_msg.seek(rank_1_commit_msg_info.start_offset)
                #     f_diff.seek(rank_1_diff_info.start_offset)
                #     rank_1_commit_msg = f_commit_msg.read(rank_1_commit_msg_info.end_offset - rank_1_commit_msg_info.start_offset)
                #     rank_1_diff = f_diff.read(min(1000, rank_1_diff_info.end_offset - rank_1_diff_info.start_offset))
                #     print('rank 1 commit msg:', rank_1_commit_msg)
                #     # print('rank 1 diff:', rank_1_diff)

                #     print('rank:', rank)
                #     import pdb; pdb.set_trace()  # DEBUG, bad cases

    for _ in output_res:
        print(_)

    # import pdb; pdb.set_trace()  # DEBUG, score_details
    with open(f"tfidf_data/tfidf_detail/scores_{owner}@@{repo}.txt", "w") as f:
        f.write("\n".join(score_details) + "\n")

    return output_res




if __name__ == "__main__":
    cve2ref_df = pd.read_csv('processed_data/valid_list.csv', header=0)
    cve2ref = dict()
    for idx, row in cve2ref_df.iterrows():
        cve = row['cve']
        if cve not in cve2ref:
            cve2ref[cve] = []   
        cve2ref[cve].append(row['patch'])


    cve2startend = json.load(open("./processed_data/cve2cand_date.json", "r"))
    all_part2tfidfdict_commit = Counter()
    all_part2tfidfdict_diff = Counter()
    numprocess = 5
    repo2startend = {}
    global linecount
    linecount = 0

    with open('./data/cve2desc.json', 'r') as f:
        cve2desc = json.load(f)

    for cve, startend in cve2startend.items():
        cve, owner, repo = cve.split("@@")
        commit_path = f'./data/commits/commits_{owner}@@{repo}.txt'
        diff_path = f'./data/diff/diff_{owner}@@{repo}.txt'
        if not os.path.exists(commit_path) or not os.path.exists(diff_path):
            continue
        if cve not in cve2desc or cve not in cve2ref:
            continue
        processed_startend = []
        for each_startend in startend:
            starttime = parse_datetime(each_startend[0])
            endtime = parse_datetime(each_startend[1])
            processed_startend.append((starttime, endtime))
        # if cve != 'CVE-2021-21321':
        #     continue
        if owner + "@@" + repo not in repo2startend:
            repo2startend[owner + "@@" + repo] = [(cve, cve2desc[cve], cve2ref[cve], processed_startend)]
        else:
            repo2startend[owner + "@@" + repo] += [(cve, cve2desc[cve], cve2ref[cve], processed_startend)]

    print('count of repo2startend:', len(repo2startend))
    # import pdb; pdb.set_trace()  # DEBUG count of repo2startend
    allitems = sorted(list(repo2startend.items()), key = lambda x:x[0]) # [100:]
    import random
    random.shuffle(allitems)

    linecount = 0

    # import pdb; pdb.set_trace()  # DEBUG#
    total_output_res = []
    

    # # # # single process
    # for item in tqdm.tqdm(allitems, total=len(allitems), desc="Cloning repositories and extracting commit messages"):
    #     output_res = get_tfidf(item)
    #     total_output_res += output_res
    #     pd.DataFrame(total_output_res, columns=['cve', 'repo', 'owner', 'patch', 'rank']).to_csv('tfidf_data/tfidf_res.csv', index=False)

    # multiprocess
    pool = Pool(processes=numprocess)
    
    cur_cummu = 0
    for output_res in tqdm.tqdm(pool.imap_unordered(get_tfidf, allitems), total=len(allitems), desc="Cloning repositories and extracting commit messages"):
        # output_res = get_tfidf(item)
        total_output_res += output_res
        cur_cummu += len(output_res)
        if cur_cummu > 100:
            pd.DataFrame(total_output_res, columns=['cve', 'repo', 'owner', 'patch', 'rank']).to_csv('tfidf_data/tfidf_res.csv', index=False)
            cur_cummu = 0

    pd.DataFrame(total_output_res, columns=['cve', 'repo', 'owner', 'patch', 'rank']).to_csv('tfidf_data/tfidf_res.csv', index=False)


