import pandas as pd
import os
import json
from tqdm import tqdm

from utils.repo_utils import parse_commit_file

if not os.path.exists('output_data'):
    os.makedirs('output_data')


def split_diff_by_file():
    pass


def load_cve_2_desc():
    with open('./data/cve2desc.json', 'r') as f:
        cve2desc = json.load(f)
    return cve2desc

def describe_stats(path = 'combine_ranker_top_100.csv'):
    cve2desc = load_cve_2_desc()
    output_jsonl_path = 'output_data/tf_idf_filtered_bert_input.json'
    output_jsonl_f = open(output_jsonl_path, 'w')

    df = pd.read_csv(path, header=0)
    for cve, cve_df in tqdm(df.groupby('cve')):
        # For ties, randomly assign integer ranks instead of fractional ranks
        # cve_df['rank'] = cve_df['scores'].rank(ascending=False, method='first').astype(int)
        # print(cve_df.head())
        if cve not in cve2desc:
            continue
        owner = cve_df['owner'].iloc[0]
        repo = cve_df['repo'].iloc[0]
        commit_msg_infos = parse_commit_file(owner, repo, 'commit_msg', [], reload=False) 
        diff_infos = parse_commit_file(owner, repo, 'diff', [], reload=False) 
        # conver format
        commit_msg_infos = {item.commit_id: item for item in commit_msg_infos}
        diff_infos = {item.commit_id: item for item in diff_infos}

        commit_file = open(f'data/commits/commits_{owner}@@{repo}.txt', 'r', encoding='iso-8859-1')
        diff_file = open(f'data/diff/diff_{owner}@@{repo}.txt', 'r', encoding='iso-8859-1')

        output_cur_json_line = dict()
        skip_commits = []
        for commit_id in cve_df['commit_id']:
            if commit_id not in commit_msg_infos or commit_id not in diff_infos:
                skip_commits.append(commit_id)
        # 'cve', 'repo', 'owner
        cve_df = cve_df[~cve_df['commit_id'].isin(skip_commits)]
        output_cur_json_line['cve'] = cve
        output_cur_json_line['repo'] = repo
        output_cur_json_line['owner'] = owner
        commit_ids, scores, labels = cve_df['commit_id'].tolist(), cve_df['scores'].tolist(), cve_df['is_patch'].tolist()
        output_cur_json_line['filtered_candidate_commits'] = commit_ids
        output_cur_json_line['filtered_scores'] = scores
        output_cur_json_line['filtered_candidate_commit_labels'] = labels
        output_cur_json_line['cve_description'] = cve2desc[cve]

        commit2str = dict()

        # Sort by scores in descending order (highest scores first)
        cve_df = cve_df.sort_values(by='scores', ascending=False)
        for idx, row in cve_df.iterrows():
            commit_id = row['commit_id']
            # import pdb; pdb.set_trace()
            commit_msg_info = commit_msg_infos[commit_id]
            diff_info = diff_infos[commit_id]
            commit_file.seek(commit_msg_info.start_offset)
            commit_msg_content = commit_file.read(commit_msg_info.end_offset - commit_msg_info.start_offset)
            diff_file.seek(diff_info.start_offset)
            diff_content = diff_file.read(diff_info.end_offset - diff_info.start_offset)

            # 
            first_newline = commit_msg_content.find('\n')
            commit_msg_content = commit_msg_content[first_newline + 1:] if first_newline != -1 else commit_msg_content
            commit_msg_content = commit_msg_content.strip()

            first_newline = diff_content.find('\n')
            diff_content = diff_content[first_newline + 1:] if first_newline != -1 else diff_content
            diff_content = diff_content.strip()

            # 去除 diff 开头的 commit_msg 
            # Remove commit message from beginning of diff if present
            if diff_content.startswith(commit_msg_content):
                diff_content = diff_content[len(commit_msg_content):].strip()

            commit2str[commit_id] = {
                'diffs': diff_content,
                'commit_msg': commit_msg_content
            }
        output_cur_json_line['commit2str'] = commit2str

        output_jsonl_f.write(json.dumps(output_cur_json_line) + '\n')

    output_jsonl_f.close()


if __name__ == "__main__":
    describe_stats()