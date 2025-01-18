# combine tfidf and time window ranker
import os
from typing import Dict, Tuple, List
import re
import pandas as pd
from functools import partial
from multiprocessing import Pool
import math
from utils.debug import *

import argparse
parser = argparse.ArgumentParser(description="Learn to rank.")
parser.add_argument(
    "--log_dis",
    action="store_true",
)
args = parser.parse_args()

def load_available_repos():
    folder_tfidf = "tfidf_data/tfidf_detail"
    folder_time_window = "tfidf_data/commit_distance_details"
    
    tfidf_pattern = re.compile(r'scores_(.*)@@(.*)\.txt')
    tfidf_files = os.listdir(folder_tfidf)
    tfidf_files = [tfidf_pattern.match(file).groups() for file in tfidf_files]
    

    commit_distance_pattern = re.compile(r'commit_distance_details_(.*)@@(.*)\.csv')
    commit_distance_files = os.listdir(folder_time_window)
    commit_distance_files = [commit_distance_pattern.match(file).groups() for file in commit_distance_files]

    # import pdb; pdb.set_trace()  # DEBUG
    time_window_files = os.listdir(folder_time_window)
    return set(commit_distance_files) & set(tfidf_files)

def load_cve_2_patch():
    cve2ref_df = pd.read_csv('processed_data/valid_list.csv', header=0)
    repo2cve_patch = dict()
    cve_set = set()
    for _, group in cve2ref_df.groupby(['owner', 'repo']):
        owner = group['owner'].values[0]
        repo = group['repo'].values[0]
        key = (owner, repo)
        if key not in repo2cve_patch:
            repo2cve_patch[key] = dict()
        
        for _, row in group.iterrows():
            cve = row['cve']
            cve_set.add(cve)
            if cve not in repo2cve_patch[key]:
                repo2cve_patch[key][cve] = []
            repo2cve_patch[key][cve].append(row['patch'])
    print(f'loaded cve count: {len(cve_set)}')
    return repo2cve_patch

def load_tfidf_res(owner, repo) -> Dict[str, List[Tuple[str, float]]]:
    '''
    result: cve -> [(commit_id, tfidf_score)]
    '''
    fpath = f"tfidf_data/tfidf_detail/scores_{owner}@@{repo}.txt"
    df = pd.read_csv(fpath, delimiter='\t', header=None)
    df.columns = ['cve', 'commit_id', 'tfidf_score', 'rank']
    
    # Verify rank is consistent with tfidf score order
    # Group by CVE and verify rank order within each CVE group
    for cve, group_df in df.groupby('cve'):
        sorted_group = group_df.sort_values('rank', ascending=True)
        for i in range(len(sorted_group)-1):
            curr_score = sorted_group.iloc[i]['tfidf_score']
            next_score = sorted_group.iloc[i+1]['tfidf_score'] 
            assert curr_score >= next_score, f"Rank order violated for CVE {cve}: rank {sorted_group.iloc[i]['rank']} score {curr_score} vs rank {sorted_group.iloc[i+1]['rank']} score {next_score}"
    
    # Convert to dictionary
    result = {}
    for _, row in df.iterrows():
        cve = row['cve']
        if cve not in result:
            result[cve] = []
        result[cve].append((row['commit_id'], row['tfidf_score']))
    return result

def load_commit_distance_res(owner, repo) -> Dict[str, List[Tuple[str, float]]]:
    fpath = f'tfidf_data/commit_distance_details/commit_distance_details_{owner}@@{repo}.csv'
    df = pd.read_csv(fpath, delimiter='\t', header=None)
    df.columns = ['cve', 'patch', 'commit_id', 'distance', 'commit_datetime']
    result = {}
    for _, row in df.iterrows():
        cve = row['cve']
        if cve not in result:
            result[cve] = []
        result[cve].append((row['commit_id'], row['distance']))
    return result

def get_combine_rank_res(owner, repo, cve2ref):
    tfidf_res = load_tfidf_res(owner, repo)
    commit_distance_res = load_commit_distance_res(owner, repo)
    
    output_res = []
    output_rank_top_100 = []
    # import pdb; pdb.set_trace()
    for cve, patch_list in cve2ref.items():
        if cve not in tfidf_res or cve not in commit_distance_res:
            import pdb; pdb.set_trace()  # DEBUG
            continue
        tfidf_details = tfidf_res[cve]
        commit_distance_details = commit_distance_res[cve]
        # Convert lists to dictionaries
        tfidf_dict = {commit_id: score for commit_id, score in tfidf_details}
        distance_dict = {commit_id: dist for commit_id, dist in commit_distance_details}

        # commit intersect
        candidate_commits = set(tfidf_dict.keys()).intersection(set(distance_dict.keys()))

        candidate_commits_2_final_scores = dict()
        for commit_id in candidate_commits:
            tfidf_score = tfidf_dict[commit_id]
            distance = distance_dict[commit_id]
            if args.log_dis:
                final_rank_score = tfidf_score * 0.5 + 1 / math.log(abs(distance) + math.e)* 0.5
            else:
                final_rank_score = tfidf_score * 0.5 + 1 / (abs(distance) + 2)* 0.5
            candidate_commits_2_final_scores[commit_id] = final_rank_score

        # get patch commit_rank
        # Sort commits by final score and get ranks
        sorted_commits = sorted(candidate_commits_2_final_scores.items(), key=lambda x: x[1], reverse=True)
        commit_ranks = {commit: rank+1 for rank, (commit, _) in enumerate(sorted_commits)}
        
        # Get ranks for patch commits
        
        
        for patch in patch_list:
            if patch not in commit_ranks:
                output_res.append((cve, repo, owner, patch, -1))
            else:
                output_res.append((cve, repo, owner, patch, commit_ranks[patch]))

            # export top 100
        for item in sorted_commits[:100]:
            # cve, repo, owner, commit_id, scores, is_patch
            output_rank_top_100.append((cve, repo, owner, item[0], item[1], item[0] in patch_list))

    
    for _ in output_res:
        print(_)
    return output_res, output_rank_top_100


if __name__ == "__main__":
    available_repos = load_available_repos()
    print('repo count: ', len(available_repos))
    cve2ref = load_cve_2_patch()
    # import pdb; pdb.set_trace()

    # process tasks
    process_tasks = []
    for owner, repo in available_repos:
        if (owner, repo) not in cve2ref:
            continue
        process_tasks.append((owner, repo, cve2ref[(owner, repo)]))
    print(len(process_tasks))
    total_output_res = []
    total_output_rank_top_100 = []
    
    # # single process
    # for owner, repo, cur_cve2ref in process_tasks:
    #     output_res, output_rank_top_100 = get_combine_rank_res(owner, repo, cur_cve2ref)
    #     total_output_res.extend(output_res)
    #     total_output_rank_top_100.extend(output_rank_top_100)

    #     # export
    #     df = pd.DataFrame(total_output_res, columns=['cve', 'repo', 'owner', 'patch', 'combined_rank'])
    #     df.to_csv('combine_ranker_res.csv', index=False)
    #     df = pd.DataFrame(total_output_rank_top_100, columns=['cve', 'repo', 'owner', 'commit_id', 'scores', 'is_patch'])
    #     df.to_csv('combine_ranker_top_100.csv', index=False)

    # multi process
    def get_combine_rank_res_wrapper(args):
        owner, repo, cur_cve2ref = args
        return get_combine_rank_res(owner, repo, cur_cve2ref)
    
    num_workers = 10
    cur_cumulation = 0
    with Pool(num_workers) as pool:
        for cur_res, output_rank_top_100 in pool.imap_unordered(get_combine_rank_res_wrapper, process_tasks):
            cur_cumulation += len(cur_res)
            total_output_res.extend(cur_res)
            total_output_rank_top_100.extend(output_rank_top_100)

            if cur_cumulation > 100:
                df = pd.DataFrame(total_output_res, columns=['cve', 'repo', 'owner', 'patch', 'combined_rank'])
                df.to_csv(f'combine_ranker_res.csv', index=False)
                df2 = pd.DataFrame(total_output_rank_top_100, columns=['cve', 'repo', 'owner', 'commit_id', 'scores', 'is_patch'])
                df2.to_csv('combine_ranker_top_100.csv', index=False)
                cur_cumulation = 0

    
    df = pd.DataFrame(total_output_res, columns=['cve', 'repo', 'owner', 'patch', 'combined_rank'])
    df.to_csv('combine_ranker_res.csv', index=False)


    df2 = pd.DataFrame(total_output_rank_top_100, columns=['cve', 'repo', 'owner', 'commit_id', 'scores', 'is_patch'])
    df2.to_csv('combine_ranker_top_100.csv', index=False)