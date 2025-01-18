#def is_valid(group_df):
import tqdm
from collections import Counter
from multiprocessing import Pool
import re
from datetime import datetime, timezone, timedelta
import json
import os
from utils.repo_utils import read_commit_file, dict_update, parse_datetime
from utils.debug import *


if not os.path.exists("data/df"):
    os.makedirs("data/df")


def is_in_range(this_time, startend_list):
    for each_startend in startend_list:
        this_start = datetime.fromisoformat(each_startend[0])
        this_end = datetime.fromisoformat(each_startend[1])
        if this_time > this_start and this_time < this_end:
            return True
    return False

def split_by_camel_snake_and_lower(sentence, stop_tokens=[]):
    sentence = re.sub("\s+", " ", sentence)
    split_pattern = r'(?<=[a-z])(?=[A-Z])|_|\s+|\.|\=|\,|\!|\?|\:|\;|\'|\"|\-|\(|\)|\{|\}|\[|\]|\>|\<|\/|\/|\#|\@|\%|\&|\*|\+|\~|\^|\`|\\'
    # camelCase -> ["camel", "case", "camelcase"], keep "camelcase"
    split_pattern2 = r'\s+|\.|\=|\,|\!|\?|\:|\;|\'|\"|\-|\(|\)|\{|\}|\[|\]|\>|\<|\/|\/|\#|\@|\%|\&|\*|\+|\~|\^|\`|\\'
    # remain special tokens, "_", "-", "."
    #split_pattern3 = r'\s+|\=|\,|\!|\?|\:|\;|\'|\"|\(|\)|\{|\}|\[|\]|\>|\<|\/|\/|\#|\@|\%|\&|\*|\+|\~|\^|\`'
    tokens = re.split(split_pattern, sentence)
    tokens = [t for t in tokens if t != '' and not t.isdigit()]
    tokens = [t.lower() for t in tokens]
    tokens2 = re.split(split_pattern2, sentence)
    tokens_set = set(tokens)
    for each_tok in tokens2:
        each_tok = each_tok.lower()
        if each_tok not in tokens_set and each_tok != '' and not each_tok.isdigit():
            tokens.append(each_tok)
    tokens_set = set(tokens)
    return tokens, tokens_set

def get_df(each_item):
    global linecount
    owner_repo, startend = each_item
    owner, repo = owner_repo.split("@@")
    print("begin", owner, repo)
  
    part2dfdict_commit = Counter()
    part2dfdict_diff = Counter()
    N_diff, N_commit = 0, 0
    N_tokens_commit, N_tokens_diff = 0, 0
    
    def df_process_file(commit_content, diff_content):
        cur_part2dfdict_commit = Counter()
        cur_part2dfdict_diff = Counter()
        N_commit, N_diff = 0, 0
        N_tokens_commit, N_tokens_diff = 0, 0
        commit_tokens, commit_tokens_set = split_by_camel_snake_and_lower(commit_content, [])
        diff_tokens, diff_tokens_set = split_by_camel_snake_and_lower(diff_content, [])
        # N_commit = 1  # singlr doc
        # N_diff = 1
        for each_token in commit_tokens_set:
            if len(each_token) >= 20: continue
            cur_part2dfdict_commit[each_token] += 1
            N_tokens_commit += 1
        for each_token in diff_tokens_set:
            if len(each_token) >= 20: continue
            cur_part2dfdict_diff[each_token] += 1
            N_tokens_diff += 1
        return cur_part2dfdict_commit, cur_part2dfdict_diff, 1, 1, N_tokens_commit, N_tokens_diff

    total_commit_d, commit_2_df_part = read_commit_file(owner, repo, {'all': startend}, df_process_file, reload=True)
    if total_commit_d is None:
        return 0, 0, 0, 0
        
    for commit, (df_part_commit, df_part_diff, cur_N_commit, cur_N_diff, cur_N_tokens_commit, cur_N_tokens_diff) in commit_2_df_part.items():
        part2dfdict_commit = dict_update(part2dfdict_commit, df_part_commit)
        part2dfdict_diff = dict_update(part2dfdict_diff, df_part_diff)
        N_commit += cur_N_commit
        N_diff += cur_N_diff
        N_tokens_commit += cur_N_tokens_commit
        N_tokens_diff += cur_N_tokens_diff



    #part2dfdict_commit = {key: part2dfdict_commit[key] for key in part2dfdict_commit if part2dfdict_commit[key] > 1}
    #part2dfdict_diff = {key: part2dfdict_diff[key] for key in part2dfdict_diff if part2dfdict_diff[key] > 1}
    # print(owner, repo)
    json.dump({'df': part2dfdict_commit, 'N': N_commit, 'token_count': N_tokens_commit}, open(f"data/df/df_{owner}@@{repo}_commit.json", "w"))
    json.dump({'df': part2dfdict_diff, 'N': N_diff, 'token_count': N_tokens_diff}, open(f"data/df/df_{owner}@@{repo}_diff.json", "w"))

    linecount += 1
    #return Counter(part2dfdict_commit), Counter(part2dfdict_diff)
    # import pdb; pdb.set_trace()
    return N_commit, N_diff, N_tokens_commit, N_tokens_diff

def merge_dict(all_part2dfdict, part2dfdict):
    for part in part2dfdict.keys():
        for word, df in part2dfdict[part].items():
            all_part2dfdict[part].setdefault(word, df)
            all_part2dfdict[part][word] += df
    return all_part2dfdict

def multiprocessing_df(allitems):
    total_N_commit, total_N_diff = 0, 0
    total_N_tokens_commit, total_N_tokens_diff = 0, 0
    for N_commit, N_diff, N_tokens_commit, N_tokens_diff in tqdm.tqdm(pool.imap(get_df, allitems), total=len(allitems), desc="Cloning repositories and extracting commit messages"):
        total_N_commit += N_commit
        total_N_diff += N_diff
        total_N_tokens_commit += N_tokens_commit
        total_N_tokens_diff += N_tokens_diff
    return total_N_commit, total_N_diff, total_N_tokens_commit, total_N_tokens_diff


def single_process_df(allitems):
    total_N_commit, total_N_diff = 0, 0
    total_N_tokens_commit, total_N_tokens_diff = 0, 0
    for each_item in tqdm.tqdm(allitems, total=len(allitems), desc="Cloning repositories and extracting commit messages"):
        N_commit, N_diff, N_tokens_commit, N_tokens_diff = get_df(each_item)
        total_N_commit += N_commit
        total_N_diff += N_diff
        total_N_tokens_commit += N_tokens_commit
        total_N_tokens_diff += N_tokens_diff

    return total_N_commit, total_N_diff, total_N_tokens_commit, total_N_tokens_diff

if __name__ == "__main__":
    cve2startend = json.load(open("./processed_data/cve2cand_date.json", "r"))
    all_part2dfdict_commit = Counter()
    all_part2dfdict_diff = Counter()
    numprocess = 5
    repo2startend = {}
    global linecount
    linecount = 0

    for cve, startend in cve2startend.items():
        cve, owner, repo = cve.split("@@")
        for each_startend in startend:
            starttime = parse_datetime(each_startend[0])
            endtime = parse_datetime(each_startend[1])
            repo2startend.setdefault(owner + "@@" + repo, [datetime.max.replace(tzinfo=timezone.utc), datetime.min.replace(tzinfo=timezone.utc)])
            repo2startend[owner + "@@" + repo][0] = min(repo2startend[owner + "@@" + repo][0], starttime)
            repo2startend[owner + "@@" + repo][1] = max(repo2startend[owner + "@@" + repo][1], endtime)

    for key in repo2startend.keys():
        repo2startend[key] = tuple(repo2startend[key])
    
    # import pdb; pdb.set_trace()
    allitems = sorted(list(repo2startend.items()), key = lambda x:x[0]) #[:30] # [100:]
    print(f'task count: {len(allitems)}')
    pool = Pool(processes=numprocess)
    linecount = 0

    # total_N_commit, total_N_diff, total_N_tokens_commit, total_N_tokens_diff = single_process_df(allitems)
    total_N_commit, total_N_diff, total_N_tokens_commit, total_N_tokens_diff = multiprocessing_df(allitems)

    with open('data/df/N_df.json', 'w') as f:
        json.dump({'N_commit': total_N_commit, 'N_diff': total_N_diff, 'N_tokens_commit': total_N_tokens_commit, 'N_tokens_diff': total_N_tokens_diff}, f)

    #for part2dfdict_commit, part2dfdict_diff in text_generator:
    #    linecount += 1
    #    if linecount % 10 == 0:
    #        print(linecount)
    #    all_part2dfdict_commit = all_part2dfdict_commit + part2dfdict_commit
    #    all_part2dfdict_diff = all_part2dfdict_diff + part2dfdict_diff
    #json.dump(dict(all_part2dfdict_commit), open("processed_data/df_commit.json", "w"))
    #json.dump(dict(all_part2dfdict_diff), open("processed_data/df_diff.json", "w"))
    # python compute_df.py 2>&1 | tee log/df1218.log
