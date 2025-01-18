import json
import requests
import shutil
from tqdm import tqdm
import os
import sys
import subprocess
import pandas as pd
from typing import Dict, Tuple, List
import re
import pickle
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

# get pareng folder
current_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(current_folder)
# import pdb; pdb.set_trace()
# 加入 path
sys.path.append(root_folder)

from utils.debug import *
import logging

if not os.path.exists('./log'):
    os.makedirs('./log')

# add logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(f"{root_folder}/log/repo_utils.log"))




def is_in_time_ranges(time_stamp: int, ranges: List[Tuple[int, int]]) -> Tuple[bool, bool]:
    '''判断一个时间是否在一系列的range中
    第一个 bool 判断的是是否在其中一个 range 中
    第二个 bool 判断的是是否 **早于** 了 range 中所有的时间范围
    '''
    if not ranges:
        return False
    # if range is Tuple
    if isinstance(ranges, tuple) and len(ranges) == 2:
        ranges = [ranges]
    in_range = False
    for start, end in ranges:
        if start <= time_stamp <= end:
            in_range = True
            break
    return in_range

def parse_datetime(this_datetime_str: str):
    # Example of ISO format: 2015-09-17T19:09:37Z or 2015-09-17T19:09:37.123Z
    if this_datetime_str[4] == "-" and "T" in this_datetime_str:  # ISO format
        if this_datetime_str.endswith("Z"):
            try:
                this_datetime = datetime.strptime(this_datetime_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                this_datetime = datetime.strptime(this_datetime_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        else:
            try:
                this_datetime = datetime.strptime(this_datetime_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                # Handle any other ISO format variations
                this_datetime = datetime.fromisoformat(this_datetime_str).replace(tzinfo=timezone.utc)
    else:  # Git log format
        try:
            # Try parsing with timezone offset
            this_datetime = datetime.strptime(this_datetime_str, "%a %b %d %H:%M:%S %Y %z")
        except ValueError:
            try:
                # Try parsing without timezone offset
                this_datetime = datetime.strptime(this_datetime_str, "%a %b %d %H:%M:%S %Y").replace(tzinfo=timezone.utc)
            except ValueError:
                # Try parsing with timezone offset at the end (+0200 format)
                datetime_parts = this_datetime_str.rsplit(" ", 1)
                base_dt = datetime.strptime(datetime_parts[0], "%a %b %d %H:%M:%S %Y")
                tz_offset = datetime_parts[1]
                hours = int(tz_offset[1:3])
                minutes = int(tz_offset[3:5])
                total_seconds = (hours * 60 + minutes) * 60
                if tz_offset.startswith("-"):
                    total_seconds = -total_seconds
                this_datetime = base_dt.replace(tzinfo=timezone(timedelta(seconds=total_seconds)))
    return this_datetime


def dict_update(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] += value
        else:
            dict1[key] = value
    return dict1


@dataclass
class CommitInfo:
    commit_id: str
    part: str  # commit_msg or diff
    datetime: int
    start_offset: int
    end_offset: int

    # sort function
    def __lt__(self, other):
        return self.datetime < other.datetime


def parse_commit_file(owner: str, repo: str, part: str, time_ranges: Tuple[datetime, datetime], reload: bool = False) -> Dict[str, CommitInfo]:
    '''
    解析 commit 文件 
    输入: time_ranges, start end datetime, 也不用解析全部的, 已有范围内足够大的就可以了
        time_ranges 需要足够大，否则会在后续的处理中漏掉一些 commits
    输出: List of (commit, datetime, 行数, 内容)
    '''
    cached_res_file = f'{root_folder}/data/commit_info_cache/{owner}@@{repo}@@{part}.pkl'
    if not os.path.exists(f'{root_folder}/data/commit_info_cache'):
        os.makedirs(f'{root_folder}/data/commit_info_cache')
    
    if os.path.exists(cached_res_file) and not reload:
        with open(cached_res_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    # parse commit file
    if part == 'commit_msg':
        commit_file_path = f'{root_folder}/data/commits/commits_{owner}@@{repo}.txt'
    elif part == 'diff':
        commit_file_path = f'{root_folder}/data/diff/diff_{owner}@@{repo}.txt'
    else:
        raise ValueError(f"Invalid part: {part}, should be commit_msg or diff")
    
    # pattern
    # if part == 'commit_msg':
    #     raw_filter = 'Commit'
    #     info_line_pattern = r'^\s*([a-f0-9]{40})\s+([A-Za-z]+\s[A-Za-z]+\s\d{1,2}\s\d{2}:\d{2}:\d{2}\s\d{4}\s[+\-]\d{4})\s*(?:.*?)\s*$'
    # else:
    raw_filter = 'Commit:'
    info_line_pattern = r"Commit:\s([a-f0-9]{40})\s+Datetime:\s+(?P<datetime>[A-Za-z].+)$"
    
    this_commit = ""
    last_commit = ""
    this_offset = 0
    last_offset = 0
    this_datetime = None
    last_datetime = None

    if not os.path.exists(commit_file_path):
        return []

    commit_infos = []
    # pybar = tqdm(desc=f"Processing file: {commit_file_path}")
    with open(commit_file_path, 'r', encoding='iso-8859-1') as f:
        while True:
            # read line and get offset
            cur_offset = f.tell()  
            line = f.readline()
            # if part == 'commit_msg':
            #     import pdb; pdb.set_trace() # DEBUG
            # pybar.update(1)
            
            if not line:  # 文件读完时readline返回空字符串
                break

            # if cur_offset > 10000000: break   # DEBUG 不读太多
            if line.strip() == "": 
                continue
            if raw_filter in line:  # 粗筛 的 str
                matches = re.search(info_line_pattern, line)
                if matches:
                    last_commit = this_commit
                    this_commit = matches.group(1)
                    last_datetime = this_datetime
                    this_datetime = parse_datetime(matches.group(2))

                    assert len(this_commit) == 40

                    if last_commit == '': continue
                    else:
                        last_offset = this_offset
                        this_offset = cur_offset
                        last_offset_range = [last_offset, this_offset]

                    # ============================================ proceess commit/diff ======================================
                    # in time range
                    if last_datetime > time_ranges[0] and last_datetime < time_ranges[1]:
                        commit_infos.append(CommitInfo(last_commit, part, last_datetime, last_offset_range[0], last_offset_range[1]))
                        # import pdb; pdb.set_trace()
                    # ============================================ proceess commit/diff ======================================     
            
        # process the last 
        # ============================================ proceess commit/diff ======================================
        if this_datetime is not None:
            if this_datetime > time_ranges[0] and this_datetime < time_ranges[1]:
                commit_infos.append(CommitInfo(this_commit, part, this_datetime, this_offset, f.tell()))
        # ============================================ proceess commit/diff ======================================

    commit_infos = sorted(commit_infos)
    with open(cached_res_file, 'wb') as f:
        pickle.dump(commit_infos, f)
    return commit_infos


def safe_zip(commit_infos, diff_infos):
    '''
    输出 commit_id, commit_info, diff_info, this_datetime 的 Generator

    如果其中的一个不存在，则输出 None
    '''
    # 生成 commit_id -> datetime 的 dict。并按照 datetime 排序
    commit_id_2_datetime = {item.commit_id: item.datetime for item in commit_infos}
    for item in diff_infos:
        commit_id_2_datetime[item.commit_id] = item.datetime
    sorted_commit_ids = sorted(commit_id_2_datetime.keys(), key=lambda x: commit_id_2_datetime[x])
    
    diff_infos_commmts = set([item.commit_id for item in diff_infos])
    diff_commit_id_2_index = {item.commit_id: index for index, item in enumerate(diff_infos)}
    commit_infos_commmts = set([item.commit_id for item in commit_infos])
    commit_commit_id_2_index = {item.commit_id: index for index, item in enumerate(commit_infos)}

    for commit_id in sorted_commit_ids:
        if commit_id in diff_infos_commmts and commit_id in commit_infos_commmts:
            yield commit_id, commit_infos[commit_commit_id_2_index[commit_id]], diff_infos[diff_commit_id_2_index[commit_id]], commit_id_2_datetime[commit_id]
        elif commit_id in diff_infos_commmts:
            yield commit_id, None, diff_infos[diff_commit_id_2_index[commit_id]], commit_id_2_datetime[commit_id]
        elif commit_id in commit_infos_commmts:
            yield commit_id, commit_infos[commit_commit_id_2_index[commit_id]], None, commit_id_2_datetime[commit_id]
        # else:
        #     yield commit_id, None, None, None


def get_max_time_range(time_ranges: List[Tuple[datetime, datetime]]):
    starts = [item[0] for item in time_ranges]
    ends = [item[1] for item in time_ranges]
    return min(starts), max(ends)

def read_commit_file(owner: str, repo: str, cve_2_time_ranges: Dict[str, List[Tuple[datetime, datetime]]], func_to_eval, reload: bool = False):
    '''
    输入: owner, repo, cve_2_time_ranges, func_to_eval
        voyage codesage tfidf 都需要得到 cve 对应的 candidates, 所以正常输入一个 cve_2_time_ranges 就可以了，计算 df 时，输入一个任意一个 key 对应全部的 time_ranges 即可
        func_to_eval, 对文本内容进行处理的函数，
            比如 voyage 
    输出: func_to_eval(content)
    '''
    commit_msg_file_path = f'{root_folder}/data/commits/commits_{owner}@@{repo}.txt'
    diff_file_path = f'{root_folder}/data/diff/diff_{owner}@@{repo}.txt'
    
    if not os.path.exists(commit_msg_file_path) or not os.path.exists(diff_file_path):
        # raise FileNotFoundError(f"File not found: {commit_msg_file_path} or {diff_file_path}")
        return None, None
    
    all_time_ranges = []
    for time_range in cve_2_time_ranges.values():
        if isinstance(time_range, tuple) and len(time_range) == 2:
            all_time_ranges.append(time_range)
        else:
            all_time_ranges.extend(time_range)
    commit_msg_infos = parse_commit_file(owner, repo, 'commit_msg', get_max_time_range(all_time_ranges), reload=reload) # DEBUG
    diff_infos = parse_commit_file(owner, repo, 'diff', get_max_time_range(all_time_ranges), reload=reload) # DEBUG

    diff_f = open(diff_file_path, 'r', encoding='iso-8859-1')
    commit_msg_f = open(commit_msg_file_path, 'r', encoding='iso-8859-1')


    output_cve_2_content = {}
    output_commit_2_res = {}
    total = len(commit_msg_infos)
    for commit_id, commit_msg_info, diff_info, this_datetime in tqdm(safe_zip(commit_msg_infos, diff_infos), total=total, desc="Processing commits"):
        # load info and datetime
        in_range = False
        for cve, cur_cve_time_ranges in cve_2_time_ranges.items():
            if is_in_time_ranges(this_datetime, cur_cve_time_ranges):
                in_range = True
                if cve not in output_cve_2_content:
                    output_cve_2_content[cve] = []
                output_cve_2_content[cve].append(commit_id)
        if not in_range: continue

        # read content
        if commit_msg_info is not None:
            commit_msg_f.seek(commit_msg_info.start_offset)
            commit_msg_content = commit_msg_f.read(commit_msg_info.end_offset - commit_msg_info.start_offset)

        else:
            commit_msg_content = ''
        
        if diff_info is not None:
            diff_f.seek(diff_info.start_offset)
            diff_content = diff_f.read(diff_info.end_offset - diff_info.start_offset)  # read the diff content
            if not diff_content.startswith('Commit:'):
                logger.critical(f"diff_content is wrong: {diff_content}")
                diff_content = ''
        else:
            diff_content = ''

        # import pdb; pdb.set_trace()
        # 可以在上面把 log level 设置为 debug，然后就可以看到 commit_msg_content 和 diff_content 了，是否和 commit id 对上了
        logger.debug(f"commit_id: {commit_id}\ncommit_msg_content: {commit_msg_content}\ndiff_content: {diff_content}")
        # import pdb; pdb.set_trace()   # DBEUG check 读到的内容是否是正确的
        # extract commit msg
        first_newline = commit_msg_content.find('\n')
        commit_msg_content = commit_msg_content[first_newline + 1:] if first_newline != -1 else commit_msg_content
        commit_msg_content = commit_msg_content.strip()
        # commit_msg = re.search(pattern, commit_msg_content, re.DOTALL)
        # if commit_msg:
        #     commit_msg_content = commit_msg.group(1)
        #     commit_msg_content = commit_msg_content.strip()
        # else:
        #     logger.critical(f"commit_msg_content is wrong: {commit_msg_content}")
        #     commit_msg_content = ''
        # extract diff
        # find the fisrt '\n'
        first_newline = diff_content.find('\n')
        diff_content = diff_content[first_newline + 1:] if first_newline != -1 else diff_content
        diff_content = diff_content.strip()
        
        # # rm commit msg in front of diff
        # if diff_content.startswith(commit_msg_content):
        #     diff_content = diff_content[len(commit_msg_content):].strip()
        # import pdb; pdb.set_trace()
        # if commit_id != 'dea227dda606900cc01870d08541b4dcc69d3889':
        #     continue
        # else:
        # import pdb; pdb.set_trace()
        output_commit_2_res[commit_id] = func_to_eval(commit_msg_content, diff_content)

    return output_cve_2_content, output_commit_2_res

if __name__ == "__main__":

    time_ranges = (datetime(2019, 8, 14, 21, 50, 58, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc))

    # TEST 1
    owner = "9001"
    repo = "copyparty"
    part = "commit_msg"
    

    # # TEST 2
    # # owner = "1Panel-dev"
    # # repo = "1Panel"
    # # part = "diff"

    def process_func(commit_msg_content, diff_content):
        return commit_msg_content + '\n' + diff_content
    
    
    # # parse_commit_file(owner, repo, part, time_ranges)   # func_to_eval 在这里暂时不需要
    # read_commit_file(owner, repo, {'cve_': time_ranges}, func_to_eval=process_func)




    # # test commit file parse
    # for f in os.listdir(f'{root_folder}/data/commits'):
    #     if f.endswith('.txt'):
    #         pattern = r'commits_(.+?)@@(.+?)\.txt'
    #         owner, repo = re.match(pattern, f).groups()
    #         print('owner: ', owner, 'repo: ', repo)
    #         parse_commit_file(owner, repo, 'commit_msg', time_ranges)



    # test diff file parse
    import time
    start_time = time.time()
    for f in os.listdir(f'{root_folder}/data/diff'):
        if f.endswith('.txt'):
            pattern = r'diff_(.+?)@@(.+?)\.txt'
            owner, repo = re.match(pattern, f).groups()
            print('owner: ', owner, 'repo: ', repo)
            read_commit_file(owner, repo, {'cve_': time_ranges}, func_to_eval=process_func)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
