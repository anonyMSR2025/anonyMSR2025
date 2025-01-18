from nltk.corpus import stopwords
import json
import os
from tqdm import tqdm
import math
from utils.debug import *

stopwords = set(stopwords.words('english'))

if not os.path.exists("tfidf_data/inrepo_idf"):
    os.makedirs('tfidf_data/inrepo_idf')

merged_df_diff = dict()
merged_df_commit = dict()
with open('./data/df/N_df.json', 'r') as f:
    N_df = json.load(f)
    N_diff = N_df['N_diff']
    N_commit = N_df['N_commit']
    N_tokens_diff = N_df['N_tokens_diff']
    N_tokens_commit = N_df['N_tokens_commit']

avg_doc_len_diff = N_tokens_diff / N_diff
avg_doc_len_commit = N_tokens_commit / N_commit


for file in tqdm(os.listdir('./data/df')):
    if file.endswith('diff.json'):
        with open(f'./data/df/{file}', 'r') as f:
            df_data = json.load(f)
        # import pdb; pdb.set_trace()
        for token, df in df_data['df'].items():
            if token not in merged_df_diff:
                merged_df_diff[token] = df
            else:
                merged_df_diff[token] += df
        
        
    elif file.endswith('commit.json'):
        with open(f'./data/df/{file}', 'r') as f:
            df_data = json.load(f)
        for token, df in df_data['df'].items():
            if token not in merged_df_commit:
                merged_df_commit[token] = df
            else:
                merged_df_commit[token] += df
    else:
        # import pdb; pdb.set_trace()
        continue
    
    inrepo_idf = dict()
    N = df_data['N']
    avg_doc_len = df_data['token_count'] / df_data['N']
    for token, df in tqdm(df_data['df'].items(), desc='compute inrepo idf', total=len(df_data['df'])):
        if df < 2: continue
        if token in stopwords: continue
        inrepo_idf[token] = math.log((N - df + 0.5) / (df + 0.5)) 
    idf_file_name = file.replace('df_', 'idf_')
    with open(f'./tfidf_data/inrepo_idf/{idf_file_name}', 'w') as f:
        json.dump(inrepo_idf, f)

    

print(len(merged_df_diff))
# compute idf
merged_idf_diff = dict()
merged_idf_commit = dict()

for token, df in tqdm(merged_df_diff.items(), desc='compute diff idf', total=len(merged_df_diff)):
    if df < 5: continue
    if token in stopwords: continue
    merged_idf_diff[token] = math.log((N_diff - df + 0.5) / (df + 0.5)) 

for token, df in tqdm(merged_df_commit.items(), desc='compute commit idf', total=len(merged_df_commit)):
    if df < 5: continue
    if token in stopwords: continue
    merged_idf_commit[token] = math.log((N_commit - df + 0.5) / (df + 0.5)) 


with open('./tfidf_data/idf_diff.json', 'w') as f:
    json.dump({'idf': merged_idf_diff, 'avg_doc_len': avg_doc_len_diff}, f)

with open('./tfidf_data/idf_commit.json', 'w') as f:
    json.dump({'idf': merged_idf_commit, 'avg_doc_len': avg_doc_len_commit}, f)



