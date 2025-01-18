import pandas as pd
import json
from multiprocessing import Pool
import os
import re
from datetime import datetime, timezone
from pandas import read_csv
import tqdm
import subprocess
from collections import Counter
import math
import argparse

from utils.repo_utils import read_commit_file, dict_update, parse_datetime, parse_commit_file, safe_zip
from compute_df import parse_datetime, split_by_camel_snake_and_lower

from utils.debug import *

parser = argparse.ArgumentParser(description="Parse window size argument.")
parser.add_argument(
    "--windowsize",
    type=int,
    default=400,
    help="Size of the time filter window."
)
parser.add_argument(
    "--task_fpath",
    type=str,
    default='processed_data/clone_and_extraction_task_list_sample.csv',
    help="Csv file of tasks."
)
parser.add_argument(
    "--mitre_rootdir",
    type=str,
    default='../../susan/mitre/mitre',
    help="mitre dataset."
)
args = parser.parse_args()



window_size = args.windowsize
mitre_rootdir = args.mitre_rootdir

def get_closest(commit2date, commit2idx, cve_reserve_time):
    closest_commit_reserve = None
    closest_diff_reserve = float("inf")
    closest_idx = -1

    for commit, commit_time in commit2date.items():
        diff1 = abs((commit_time - cve_reserve_time).total_seconds())
        if diff1 < closest_diff_reserve:
            closest_diff_reserve = diff1
            closest_commit_reserve = commit
            closest_idx = commit2idx[commit]
    return closest_diff_reserve, closest_commit_reserve, closest_idx


def is_commit_in_file(commit_id, file_path):
    try:
        # Run the grep command
        result = subprocess.run(
            ['grep', commit_id, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Check if grep found a match
        if result.returncode == 0:
            return True  # Commit ID is found in the file
        else:
            return False  # Commit ID is not found
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
        return False

def extract_available_cve():
    # extract cve with at least 1 commit and 1 diff
    data = read_csv("processed_data/clone_and_extraction_task_list.csv")
    filtered_data = []
    for row in tqdm.tqdm(data.iterrows()):
        cve, owner, repo, patch = row[1]
        commit_id = patch.split("/")[-1]
        is_commit = is_commit_in_file(commit_id, f"data/commits/commits_{owner}@@{repo}.txt")
        is_diff = is_commit_in_file(commit_id, f"data/diff/diff_{owner}@@{repo}.txt")
        if is_commit and is_diff:
            filtered_data.append(row)
        else:
            print(owner, repo, commit_id)

    filtered_data = pd.DataFrame(filtered_data, columns=["cve", "owner", "repo", "patch"])
    print(len(set(filtered_data["cve"])))
    filtered_data.to_csv("processed_data/cve_validcommit.csv")


def is_valid(group_df):
    all_commits_voyage = set([])

    owner = group_df["owner"].iloc[0]
    repo = group_df["repo"].iloc[0]
    patch_list = [x.split("@@")[-1] for x in group_df["patch"]]
    group_df["patch"] = patch_list
    valid_commits = []
    valid_diff = []
    cve2candidates = {}

    # 
    total_time_ranges = (datetime(1970, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc))
    commit_info_list = parse_commit_file(owner, repo, "commit_msg", total_time_ranges, reload=True)
    diff_info_list = parse_commit_file(owner, repo, "diff", total_time_ranges, reload=True)
    
    total_commit_commmt_ids = set([item.commit_id for item in commit_info_list])
    total_diff_commit_ids = set([item.commit_id for item in diff_info_list])
    
    both_valid = list(set(patch_list).intersection(total_diff_commit_ids).intersection(set(total_commit_commmt_ids)))

    commit2date = {item.commit_id: item.datetime for item in commit_info_list}
    commit_list = [item.commit_id for item in commit_info_list]    
    for item in diff_info_list:
        commit2date[item.commit_id] = item.datetime
        commit_list.append(item.commit_id)
    
    commit_list = list(set(commit_list))


    # import pdb; pdb.set_trace()
    filtered_groupdf = group_df[group_df.patch.isin(both_valid)]
    filtered_groupdf2 = []

    commit_distance_info = []
    for _, row in filtered_groupdf.iterrows():
        this_cve, owner, repo, patch, sources = row
        #this_cve = filtered_groupdf["cve"].iloc[idx]
        cve_folder_name = str(this_cve.split("-")[-1])[:-3] + "xxx"
        # CVE-2010-1666
        year = str(this_cve.split("-")[1])
        if not os.path.exists(f"{mitre_rootdir}/cvelistV5-main/cves/" + year + "/" + cve_folder_name + "/" + this_cve + ".json"): 
            return None
        cve_metadata = json.load(open(f"{mitre_rootdir}/cvelistV5-main/cves/" + year + "/" + cve_folder_name + "/" + this_cve + ".json"))["cveMetadata"]
        if "dateReserved" not in cve_metadata: 
            continue
        
        date_reserve_str = cve_metadata["dateReserved"]
        if date_reserve_str.endswith("Z"):
            try:
                cve_reserve_time = datetime.strptime(date_reserve_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                cve_reserve_time = datetime.strptime(date_reserve_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        else:
            cve_reserve_time = datetime.strptime(date_reserve_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)

        sorted_commitlist = sorted(commit_list, key = lambda x:commit2date[x])
        commit2idx = {sorted_commitlist[idx]: idx for idx in range(len(sorted_commitlist))}

        closest_diff_reserve, closest_commit_reserve, closest_idx_reserve = get_closest(commit2date, commit2idx, cve_reserve_time)
        # Store commit id, patch distance and datetime info
        
        for commit_id in sorted_commitlist:
            commit_idx = commit2idx[commit_id]
            distance = commit_idx - closest_idx_reserve
            commit_datetime = commit2date[commit_id].isoformat()
            commit_distance_info.append([this_cve, patch, commit_id, distance, commit_datetime])
        # import pdb; pdb.set_trace()  # DEBUG commit_distance_info
      
      
        left_idx = max(closest_idx_reserve - window_size, 0)
        right_idx = min(closest_idx_reserve + window_size, len(sorted_commitlist) - 1)
        left_time = commit2date[sorted_commitlist[left_idx]].isoformat()
        right_time = commit2date[sorted_commitlist[right_idx]].isoformat()
        assert left_time < right_time
        # for cand_idx in range(max(closest_idx_reserve - 150, 0), min(closest_idx_reserve + 150, len(commit_list))):
        #    all_commits_voyage.add(commit_list[cand_idx])
        cve2candidates.setdefault(this_cve + "@@" + owner + "@@" + repo, [])
        cve2candidates[this_cve + "@@" + owner + "@@" + repo].append([left_time, right_time])

        this_patch_idx = commit2idx[patch]
        filtered_groupdf2.append(list(row) + [this_patch_idx - closest_idx_reserve])
        
    # STORE commit_distance_info
    if not os.path.exists("tfidf_data/commit_distance_details"):
        os.makedirs("tfidf_data/commit_distance_details", exist_ok=True)
    with open(f"tfidf_data/commit_distance_details/commit_distance_details_{owner}@@{repo}.csv", "w") as f:
        for row in commit_distance_info:
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\n")
    # import pdb; pdb.set_trace()
    return filtered_groupdf2, cve2candidates, all_commits_voyage


def merge_dict(dict1, dict2):
    for key, val in dict2.items():
        dict1.setdefault(key, [])
        dict1[key] = dict1[key] + val
    return dict1
        
if __name__ == "__main__":
    gt_data_df = pd.read_csv(args.task_fpath, header=0)
    print(len(gt_data_df))

    commit_repo_list = [file.rstrip()[8:-4] for file in os.listdir("./data/commits/")]
    diff_repo_list = [file.rstrip()[5:-4] for file in os.listdir("./data/diff/")]

    filtered_idx = [x for x in range(len(gt_data_df)) if gt_data_df["owner"].iloc[x] + "@@" + gt_data_df["repo"].iloc[x] in set(commit_repo_list).intersection(diff_repo_list)]
    gt_data_df = gt_data_df.iloc[filtered_idx]

    group_df_list = []
    numprocess = 5
    pool = Pool(processes=numprocess)
    for (owner, repo), group_df in gt_data_df.groupby(['owner', 'repo']):
        # if owner == "theupdateframework" and repo == "notary":
            # import pdb; pdb.set_trace()
        group_df_list.append(group_df)
        # else:
            # pass
    linecount = 0
    output_data = []
    all_cve2cand = {}
    total_commits = set([])
    # multi process
    for cur_repo_output_data in tqdm.tqdm(pool.imap(is_valid, group_df_list), total=len(group_df_list), desc="Cloning repositories and extracting commit messages"):
    # for cur_group in tqdm.tqdm(group_df_list, total=len(group_df_list), desc="Processing group"):
        # cur_repo_output_data = is_valid(cur_group)
        if cur_repo_output_data:
            cur_repo_output_data, cve2cand, all_commits_voyage = cur_repo_output_data
            total_commits.update(all_commits_voyage)
            output_data += cur_repo_output_data
            all_cve2cand = merge_dict(all_cve2cand, cve2cand)
        linecount += 1
    output_data = pd.DataFrame(output_data, columns = ["cve", "owner", "repo", "patch", "sources", "diff_idx"])
    output_data.to_csv("./processed_data/valid_list.csv")
    for key in all_cve2cand.keys():
        all_cve2cand[key] = list(all_cve2cand[key])
    print(len(total_commits))
    json.dump(all_cve2cand, open("./processed_data/cve2cand_date.json", "w"))



