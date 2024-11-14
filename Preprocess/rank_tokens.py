import math
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
english_stopwords = set(stopwords.words("english") + [""])
import json
import multiprocessing
import os
import re
from typing import List

import pandas as pd
import tqdm
#from tqdm import tqdm
import time

def get_df(data_dir, title_repo):
    token2df = {}
    cve2desc = {}
    cve2cvedf = {}
    cve2count = {}
    t1 = time.time()
    for suffix in ["train_downsampled", "test", "valid"]:
        fin = open(os.path.join(data_dir, f"tf_idf_filtered_bert_input_{suffix}.json"), "r")
        commit = 0
        for line in fin:
            if commit % 1000 == 0:
                print(commit)
                print(time.time() - t1)
            commit += 1
            data = json.loads(line)
            cve = data["cve"]
            cve2count.setdefault(cve, 0)
            cve2count[cve] += 1
            commit_message_tokens = re.split(r"[\W_]+", data["commit_msg"].lower()) # cannot use camel/snake to split, too slow
            diff_tokens = re.split(r"[\W_]+", " ".join(data["diff"]).lower())
            for each_token in set(commit_message_tokens).union(set(diff_tokens)):
                token2df.setdefault(each_token, set([]))
                token2df[each_token].add(cve)
            cve2desc[cve] = data["cve_desc"]
            stop_tokens = title_repo[cve]["title"] + title_repo[cve]["owner_repo"].split("\/")
            commit_msg_tokens = split_by_camel_snake_and_lower(data["commit_msg"], stop_tokens).keys()
            diff_tokens = split_by_camel_snake_and_lower(" ".join(data["diff"]), stop_tokens).keys()
            commit_msg_tokens = set(commit_msg_tokens) #.intersection(cve_desc_tokens)
            diff_tokens = set(diff_tokens) #.intersection(cve_desc_tokens)
            for each_token in commit_msg_tokens.union(diff_tokens):
                if each_token in english_stopwords: continue
                cve2cvedf.setdefault(cve, {})
                cve2cvedf[cve].setdefault(each_token, 0)
                cve2cvedf[cve][each_token] += 1
    token2df_ = {}
    for token in token2df.keys():
        token2df_[token] = len(token2df[token])
    json.dump(cve2desc, open(os.path.join(data_dir, "cve2desc.json"), "w"))
    json.dump(token2df_, open(os.path.join(data_dir, "token2df.json"), "w"))
    json.dump(cve2cvedf, open(os.path.join(data_dir, "cve2cvedf.json"), "w"))
    json.dump(cve2count, open(os.path.join(data_dir, "cve2count.json"), "w"))

def tfidf_score2(cve_tokens, commit_msg_tokens, diff_tokens, token2msgtf, token2difftf, token2df, token2alldf, samecve_count, all_count):
    cve_sep = {}
    commit_msg_sep = {}
    diff_sep = {}
    for each_token in cve_tokens:
       if each_token in english_stopwords: continue
       offset = 0.01
       df1 = token2df.get(each_token, 0) + offset
       df2 = token2alldf.get(each_token, 0) + offset
       weighted_tf = 0.5 * math.log(1 + token2msgtf.get(each_token, 0), 2) + 0.5 * math.log(1 + token2difftf.get(each_token, 0), 2)
       samecve_idf = max(math.log( samecve_count / df1, 2) - 1, 0)
       all_idf = max(math.log(all_count / df2, 2) - 0.01, 0)
       if weighted_tf * (samecve_idf * all_idf) > 0:
           cve_sep[each_token] = (cve_tokens[each_token], weighted_tf * (samecve_idf * all_idf))
           if each_token in commit_msg_tokens:
              commit_msg_sep[each_token] = (commit_msg_tokens[each_token], weighted_tf * (samecve_idf * all_idf))
           if each_token in diff_tokens:
              diff_sep[each_token] = (diff_tokens[each_token], weighted_tf * (samecve_idf * all_idf))
    return cve_sep, commit_msg_sep, diff_sep


def split_by_camel_snake_and_lower(sentence, stop_tokens):
    split_pattern = r'(?<=[a-z])(?=[A-Z])|_|\s+|\.|\=|\,|\!|\?|\:|\;|\'|\"|\-|\(|\)|\{|\}|\[|\]|\>|\<|\/|\/|\#|\@|\%|\&|\*|\+|\~|\^|\`'
    # camelCase -> ["camel", "case", "camelcase"], keep "camelcase"
    split_pattern2 = r'\s+|\.|\=|\,|\!|\?|\:|\;|\'|\"|\-|\(|\)|\{|\}|\[|\]|\>|\<|\/|\/|\#|\@|\%|\&|\*|\+|\~|\^|\`'
    # remain special tokens, "_", "-", "."
    #split_pattern3 = r'\s+|\=|\,|\!|\?|\:|\;|\'|\"|\(|\)|\{|\}|\[|\]|\>|\<|\/|\/|\#|\@|\%|\&|\*|\+|\~|\^|\`'
    #tokens = re.split(split_pattern, sentence)
    matches = list(re.finditer(split_pattern, sentence)) 
    #tokens2 = re.split(split_pattern2, sentence)
    matches2 = list(re.finditer(split_pattern2, sentence))
    #tokens3 = re.split(split_pattern3, sentence)
    all_tokens = {} # + tokens2 + tokens3
    for idx in range(len(matches) + 1):
        left = 0 if idx == 0 else matches[idx - 1].span()[1]
        right = len(matches) if idx == len(matches) else matches[idx].span()[0]
        this_token = sentence[left:right]
        if this_token == "": continue
        this_token = re.sub("[^a-zA-Z]", "", this_token).lower()
        if this_token in stop_tokens: continue
        all_tokens.setdefault(this_token, [])
        all_tokens[this_token].append((left, right))
    tokens_set = set(all_tokens)
    for idx in range(len(matches2) + 1):
        left = 0 if idx == 0 else matches2[idx - 1].span()[1]
        right = len(matches2) if idx == len(matches2) else matches2[idx].span()[0]
        this_token = sentence[left:right]
        if this_token not in all_tokens:
           if this_token == "": continue
           this_token = re.sub("[^a-zA-Z]", "", this_token).lower()
           if this_token in stop_tokens: continue
           all_tokens.setdefault(this_token, [])
           all_tokens[this_token].append((left, right))
    return all_tokens

def rank_tokens(data_dir):
    title_repo = json.load(open(os.path.join(data_dir, "cve_2_porject_info.json"), "r"))

    if not os.path.exists(os.path.join(data_dir, 'token2df.json')):
        get_df(data_dir, title_repo)
    
    output_df_data = []
    cve2desc = json.load(open(os.path.join(data_dir, "cve2desc.json"), "r"))
    token2alldf = json.load(open(os.path.join(data_dir, "token2df.json"), "r"))
    cve2cvedf = json.load(open(os.path.join(data_dir, "cve2cvedf.json"), "r"))
    cve2count = json.load(open(os.path.join(data_dir, "cve2count.json"), "r"))
    #cve_set = list(cve2desc.keys())

    for suffix in ["train_downsampled", "test", "valid"]:
        fin = open(os.path.join(data_dir, f"tf_idf_filtered_bert_input_{suffix}.json"), "r")
        commit = 0
        cve_sep_list = []
        commit_msg_sep_list = []
        diff_sep_list = []
        for line in tqdm.tqdm(fin):
            if commit % 1000 == 0:
                print(commit)
            commit += 1
            data = json.loads(line)
            cve = data["cve"]
            #commit_message_tokens = re.split(r"[\W_]+", data["commit_msg"].lower()) # cannot use camel/snake to split, too slow
            diff_tokens = re.split(r"[\W_]+", " ".join(data["diff"]).lower())
            stop_tokens = title_repo[cve]["title"] + title_repo[cve]["owner_repo"].split("\/")
            cve_desc_tokens = split_by_camel_snake_and_lower(data["cve_desc"], stop_tokens)
            commit_msg_tokens = split_by_camel_snake_and_lower(data["commit_msg"], stop_tokens)
            diff_tokens = split_by_camel_snake_and_lower(" ".join(data["diff"]), stop_tokens)
            cvedf = cve2cvedf.get(cve, {})
            commit_msg_token2tf = {}
            diff_token2tf = {}
            for each_token in commit_msg_tokens:
                if each_token in cve_desc_tokens:
                   commit_msg_token2tf.setdefault(each_token, 0)
                   commit_msg_token2tf[each_token] += len(commit_msg_tokens[each_token])
            for each_token in diff_tokens:
                if each_token in diff_tokens:
                    diff_token2tf.setdefault(each_token, 0)
                    diff_token2tf[each_token] += len(diff_tokens[each_token])
            
            cve_sep, commit_msg_sep, diff_sep = tfidf_score2(cve_desc_tokens, commit_msg_tokens, diff_tokens, commit_msg_token2tf, diff_token2tf, cvedf, token2alldf, cve2count[cve] + 10, 3500)       
            cve_sep_list.append(cve_sep)
            commit_msg_sep_list.append(commit_msg_sep)
            diff_sep_list.append(diff_sep)
        json.dump({"cve_sep": cve_sep_list, "commit_msg_sep": commit_msg_sep_list, "diff_sep": diff_sep_list}, open(os.path.join(data_dir, f"{suffix}_sep_2.json"), "w"))
            
if __name__ == '__main__':
    
    # cve = 'CVE-2021-46822'
    # owner_project = ["curl", 'curl']
    main()

    # CVE-2021-46822 ["libjpeg", "turbo"]
    # CVE-2022-0945 ["star7th", "showdoc"]
    # CVE-2021-29136 ["open", "containers", "opencontainers", "umoci"]
    # CVE-2011-1949 ["plone", "products", "portal", "transforms"]
    # CVE-2012-2691 ["mantisbt", "mantis", "bt"]
    # CVE-2012-6121 ["roundcube", "webmail"] # requires to get the title on github README
    # CVE-2013-0285 ["savonrb", "nori"]
    # CVE-2010-2272, algo output: ["dojo", "iframe", "history"]
        # project project:  dojo/dojo
        # "iframe_history.html" -> FileName ; 
    # CVE-2010-4159, algo output: ["mono"]
        # project project:  mono/mono
        # "mono" -> FolderName ; "metadata/loader.c" -> FileName
    # CVE-2011-1949, algo output: ["querystring"]
        # project project:  forkcms/forkcms
        # "querystring" -> Variable ;
    # CVE-2012-6696, []
        # project project: inspircd/inspircd
        # None
    # CVE-2013-1944, ["libcurl", "cookies"]
        # project project: curl/curl
        # "cookie.c" -> FileName ; "tailMatch" -> FunctionName ;
    # CVE-2013-2006, ["keystone", "ldap"]
        # project project: openstack/keystone
        # "ldap" -> Object ;
    # CVE-2013-4287, ['backtracking', 'gems', '0p247', 'rubygems', 'algorithmic', 'attackers', 'denial', 'consumption', 'complexity', 'crafted', 'cpu', 'rb', 'gem', 'pattern', 'expression', 'ruby', 'regular', 'vulnerability', '23']
        # project project: rubygems/rubygems
        # False Positive: rb, 0p247, rubygems, algorithmic, attackers, denial, consumption ??
    # CVE-2014-0034, ['sts', 'saml']
        # project project: apache/cxf
    

