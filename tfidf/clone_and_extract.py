import json
from pandas import read_csv
import requests
import shutil
from tqdm import tqdm
import os
import subprocess
import pandas as pd
from typing import Dict, Tuple, List
from dateutil import parser
import re
import pickle
import argparse

from utils.debug import *

parser = argparse.ArgumentParser(description="Clone repo and extract commit messages and diffs.")
parser.add_argument(
    "--task_fpath",
    type=str,
    default='processed_data/clone_and_extraction_task_list_sample.csv',
    help="Csv file of tasks."
)
args = parser.parse_args()


if not os.path.exists('data/commits_cache'):  # cache 解析后的 repo 的 commit data
    os.makedirs('data/commits_cache')

if not os.path.exists('data/commits'): 
    os.makedirs('data/commits')

if not os.path.exists('data/diff'): 
    os.makedirs('data/diff')

home_dir = os.path.expanduser("~")
github_token = json.load(open(home_dir + "/secret.json", "r"))["github"]

def can_clone_github_repo(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"Bearer {github_token}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("private") is False  # 如果是公有仓库，则可以克隆
        print("return status wrong")
        return False
    except requests.exceptions.RequestException:
        return False

def clone_or_pull_repo(owner, repo, remove_repo: bool = False):
    repo_path = os.path.join("repo", f"{owner}@@{repo}")
    commits_output_path = os.path.join("data/commits", f"commits_{owner}@@{repo}.txt")
    diff_output_path = os.path.join("data/diff", f"diff_{owner}@@{repo}.txt")
    # import pdb; pdb.set_trace()  
    if os.path.exists(commits_output_path) and os.path.exists(diff_output_path):
       print(f"Skipping {owner}/{repo} as it has already been processed.")
       return repo_path
    try:
        if not os.path.exists(repo_path):
            print(f"Cloning {repo} as a bare repository...")
            subprocess.run(
                ["git", "clone", f"https://github.com/{owner}/{repo}.git", repo_path],
                check=True
            )
            os.chdir(repo_path)
            fout = open(f"../../data/commits/commits_{owner}@@{repo}.txt", "w")
            subprocess.run(["git", "log", "--pretty=format:Commit: %H Datetime: %ad\n %B", "--date-order", "--reverse"], stdout=fout, stderr=subprocess.PIPE, text = True)
            fout.close()
            # import pdb; pdb.set_trace()
            fout = open(f"../../data/diff/diff_{owner}@@{repo}.txt", "w")
            subprocess.run(["git", "log", "-p", "--pretty=format:Commit: %H Datetime: %ad\n %s", "--author-date-order", "--reverse"], stdout=fout, stderr=subprocess.PIPE, text = True)
            # fout.close()
            os.chdir("../../")  # move back to the root directory
        else:
            print(f"Fetching latest changes for {repo}...")
            subprocess.run(["git", "-C", repo_path, "fetch", "--all"], check=True)
            # pass  #  no need to fetch
        
        # if commit_id is not None:  # output commit msg
        #     if not os.path.exists(f'./data/commit_msg/{owner}@@{repo}'):
        #         os.makedirs(f'./data/commit_msg/{owner}@@{repo}')
        #     otuput_f = f"../../data/commit_msg/{owner}@@{repo}/{commit_id}.txt"
        #     os.chdir(repo_path)
        #     if not os.path.exists(otuput_f):
        #         fout = open(otuput_f, "w")
        #         subprocess.run(["git", "log", "-1", "--format=%B", commit_id], stdout=fout, stderr=subprocess.PIPE, text=True)
        #         fout.close()
        #     os.chdir("../../")
        if remove_repo:
            shutil.rmtree(repo_path)
        return repo_path
    
    except subprocess.CalledProcessError:
        return None


def safe_read_lines(file_path: str) -> List[str]:
    """
    安全地读取文件内容，尝试多种编码方式。
    如果所有编码方式都失败，返回空列表。
    
    Args:
        file_path: 文件路径
    Returns:
        文件内容的行列表，如果读取失败则返回空列表
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file {file_path} with {encoding}: {str(e)}")
            continue
    
    print(f"Warning: Failed to read {file_path} with all encodings, returning empty list")
    return []


def parse_commit_time(owner: str, repo: str, reload: bool = False) -> Dict[str, Tuple[int, str]]:
    '''
    输入一个 repo 的 commits 的文件，输出一个 dict，key 是 commit_id，value 是 (commit_time, commit_msg)
    '''
    commit_time_cache_file_path = os.path.join("data/commits_cache", f"{owner}@@{repo}.pkl")
    
    if not reload and os.path.exists(commit_time_cache_file_path):
        try:
            with open(commit_time_cache_file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Error loading cache file {commit_time_cache_file_path}: {str(e)}")
            

    commit_time_file_path = os.path.join("data/commits", f"commits_{owner}@@{repo}.txt")
    # with open(commit_time_file_path, 'r') as f:
    file_lines = safe_read_lines(commit_time_file_path)
    commit_time_data = {}
    current_commit = None
    commit_msg = []
    
    for line in file_lines:
        line = line.strip()
        if re.match(r'^commit [0-9a-f]+$', line):
            # Save previous commit data if exists
            if current_commit is not None:
                commit_time_data[current_commit] = (commit_timestamp, '\n'.join(commit_msg).strip())
            
            # Start new commit
            # import pdb; pdb.set_trace()
            current_commit = line.split(' ')[1]
            commit_msg = []
            commit_timestamp = None
            
        elif line.startswith('Date:'):
            # Parse date line
            date_str = ' '.join(line.split()[1:])
            try:
                commit_timestamp = int(parser.parse(date_str).timestamp())
            except Exception as e:
                print(f"Error parsing date {date_str}: {str(e)}")
                commit_timestamp = None
        elif line.startswith('Merge:'):
            pass
        elif not line.isspace() and not line.startswith('Author:'):
            # Add non-empty lines that aren't author to commit message
            commit_msg.append(line)
    
    # Add the last commit
    if current_commit is not None:
        commit_time_data[current_commit] = (commit_timestamp, '\n'.join(commit_msg))
    
    with open(commit_time_cache_file_path, 'wb') as f:
        pickle.dump(commit_time_data, f)
    return commit_time_data



if __name__ == "__main__":
  

    data_df = read_csv(args.task_fpath) 
    finished = [] #line[5:-4] for line in os.listdir("data/diff/")] #line[8:-4] for line in os.listdir("data/commits/")] #x.rstrip("\n")[5:-4] for x in open("./data/diff/finished.text", "r").readlines()]

    #data_df = [line[8:-4] for line in raw_data_df] 
    #data_df = []

    #for line in raw_data_df:
    #    print(line)
    #    if not os.path.exists(f"./data/diff/diff_{line}.txt"):
    #        data_df.append(line)
    
    from multiprocessing import Pool

    def process_group(group_data):
        owner, repo = group_data.split("@@")
        if can_clone_github_repo(owner, repo):
           clone_or_pull_repo(owner, repo, remove_repo=True)
            # else:
            #     clone_or_pull_repo(owner, repo, commit_id=commit_id)

    def remove_large_repo(group_data):
        pass
        # TODO  需要修改，根据 regex 获取到的 commit 数量来判断
        # owner, repo = group_data.split("@@")
        # if os.path.exists(f"data/commits/commits_{owner}@@{repo}.txt"):
        #     fin = open(f"data/commits/commits_{owner}@@{repo}.txt", "r", encoding="iso-8859-1")
        #     if len(fin.readlines()) > 5000:
        #         os.remove(f"data/commits/commits_{owner}@@{repo}.txt")
        #         if os.path.exists(f"data/diff/diff_{owner}@@{repo}.txt"):
        #             os.remove(f"data/diff/diff_{owner}@@{repo}.txt")


    # Create groups data
    #groups_data = sorted([line.rstrip() for line in data_df], reverse=True)
    groups_data = []
    for idx, row in data_df.iterrows():
        owner, repo = row["owner"], row["repo"]
        # if can_clone_github_repo(owner, repo):
        groups_data.append(f"{owner}@@{repo}")

    groups_data = list(set(groups_data))
    import random
    random.shuffle(groups_data)
    # # Use multiprocessing pool
    with Pool(processes=15) as pool:
        for _ in tqdm(pool.imap(process_group, groups_data), total=len(groups_data), desc="Cloning repositories"):
            pass

    
    # # single process
    # for group_data in tqdm(groups_data, desc="Cloning repositories"):
    #     process_group(group_data)
