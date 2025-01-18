import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Set, List
import os
import json
from debug import *
import pickle
from main import extract_owner_repo
from tqdm import tqdm
from dateutil import parser
from repo_utils import parse_commit_time
import multiprocessing

def load_cve_creation_time(datafile_path: str = '/workspace/filesystem/susan/mitre/mitre/cvelistV5-main') -> Dict[str, datetime]:
    cve_creation_time = {}
    
    # walk through the datafile_path folder
    for root, dirs, files in os.walk(datafile_path):
        for file in files:
            if file.endswith('.json') and file.startswith('CVE'):
                print(f"Processing file: {file}")  # TODO check is this cve files if this is the correct file
                with open(os.path.join(root, file), 'r') as f:
                    # import pdb; pdb.set_trace()
                    data = json.load(f)
                    cve_id = data['cveMetadata']['cveId']
                    try:
                        creation_time = data['cveMetadata']['datePublished']
                    except KeyError:
                        if 'dateReserved' in data['cveMetadata']:
                            creation_time = data['cveMetadata']['dateReserved']
                        else:
                            continue
                    cve_creation_time[cve_id] = parser.isoparse(creation_time)
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")
    cve_creation_time_str = {k: v.isoformat() for k, v in cve_creation_time.items()}
    with open("processed_data/cve_creation_time.json", "w") as f:
        json.dump(cve_creation_time_str, f)
    return cve_creation_time


def time_window_filter(repo_commit_time: Dict[str, Tuple[int, str]], time_window: Tuple[int, int]) -> Set[str]:
    '''
    输入
     repo_commit_time: repo 的 所有 commit hash 和commit时间
     time_window: time window
     时间都是 timestamp

    输出
        time window 内的 commit hash 的 set
    '''
    # return set(repo_commit_time.keys())  # return all commit hash, DEBUG
    commit_hash_set = set()
    for commit_hash, (commit_time, commit_msg) in repo_commit_time.items():
        if commit_time is None:
            continue
        if commit_time >= time_window[0] and commit_time <= time_window[1]:
            commit_hash_set.add(commit_hash)
    return commit_hash_set


def single_process_time_window_filter(cve_id: str, cve_2_created_time: datetime, owner: str, repo: str, hashes: List[str], time_window: Tuple[int, int]):
    # hashes = [item.split('/')[-1] for item in gt_links_l]
    repo_commit_time = parse_commit_time(owner, repo)
    cve_creation_time = cve_2_created_time
    cve_creation_time_window = [cve_creation_time - timedelta(days=-time_window[0]), cve_creation_time + timedelta(days=time_window[1])]
    cve_creation_timestamp_window = [int(cve_creation_time_window[0].timestamp()), int(cve_creation_time_window[1].timestamp())]
    commit_hash_set = time_window_filter(repo_commit_time, cve_creation_timestamp_window)
    recall = len(commit_hash_set & set(hashes)) / len(set(hashes))
    return cve_id, recall, commit_hash_set

def single_process_time_window_filter_wrapper(task: Tuple[str, datetime, str, str, List[str], Tuple[int, int]]):
    return single_process_time_window_filter(*task)

def main_time_window_filter(time_window: Tuple[int, int]):
    '''
    time window, 以天为单位， time_window[0] 是相对于creation_time 的开始时间， time_window[1] 是相对于creation_time 的结束时间
    '''
    retrieved_cve_2_commit_set = dict()
    cve_l = []
    recall_l = []
    # Step 1, load dataset
    with open('processed_data/cve_creation_time.pkl', 'rb') as f:
        cve_2_creation_time = pickle.load(f)

    # load gt
    with open('processed_data/cve_id_2_patch_cleaned.json', 'r') as f:
        cve_gt = json.load(f)

    # import pdb; pdb.set_trace()
    # Step 2, retrieve by time window
    task_l = []
    for cve_id, gt_links_l in tqdm(cve_gt.items(), desc=f"laod tasks"):
        if len(gt_links_l) == 0:
            continue
        hashes = [item.split('/')[-1] for item in gt_links_l]
        owner, repo = extract_owner_repo(gt_links_l[0])
        if owner is None or repo is None:
            continue
        commits_time_path = os.path.join("data/commits", f"commits_{owner}@@{repo}.txt")
        if not os.path.exists(commits_time_path):
            continue
        try:
            cve_creation_time = parser.isoparse(cve_2_creation_time[cve_id])
        except KeyError:  # 如果 cve_id 不在 cve_2_creation_time 中，则跳过
            continue
        task_l.append((cve_id, cve_creation_time, owner, repo, hashes, time_window))

    pool = multiprocessing.Pool(processes=10)
    for cve_id, recall, commit_hash_set in tqdm(pool.imap_unordered(single_process_time_window_filter_wrapper, task_l), desc=f"Retrieving by time window: {time_window[0]} - {time_window[1]} days", total=len(task_l)):
        cve_l.append(cve_id)
        recall_l.append(recall)
        retrieved_cve_2_commit_set[cve_id] = commit_hash_set
        # if len(cve_l) > 200:
        #     break

    print(f'time window: {time_window[0]} - {time_window[1]} days')
    print(f'avg recall: {sum(recall_l) / len(recall_l)}')   
    print(f'avg commits: {sum(len(commit_hash_set) for commit_hash_set in retrieved_cve_2_commit_set.values()) / len(retrieved_cve_2_commit_set)}')

    # for cve_id, owner, repo, gt_links_l, time_window, recall, commit_hash_set in results:
    #     repo_commit_time = parse_commit_time(owner, repo)
    #     cve_creation_time = parser.isoparse(cve_2_creation_time[cve_id])
    #     cve_creation_time_window = [cve_creation_time - timedelta(days=time_window[0]), cve_creation_time + timedelta(days=time_window[1])]
    #     cve_creation_timestamp_window = [int(cve_creation_time_window[0].timestamp()), int(cve_creation_time_window[1].timestamp())]
        
    #     # load repo_commit_time
    #     commit_hash_set = time_window_filter(repo_commit_time, cve_creation_timestamp_window)

    #     recall = len(commit_hash_set & set(hashes)) / len(set(hashes))
    #     cve_l.append(cve_id)
    #     recall_l.append(recall)
    #     if len(cve_l) > 200:
    #         break

    # print(f'time window: {time_window[0]} - {time_window[1]} days')
    # print(f'avg recall: {sum(recall_l) / len(recall_l)}')

if __name__ == "__main__":
    main_time_window_filter([0, 100])

    main_time_window_filter([-500, 500])

    main_time_window_filter([-500, 1000])
    # 
    main_time_window_filter([-500, 3000])
    
    # main_time_window_filter([0, 4000])

    # main_time_window_filter([0, 5000])

    # main_time_window_filter([0, 6000])

    # main_time_window_filter([0, 7000])

    # main_time_window_filter([0, 8000])