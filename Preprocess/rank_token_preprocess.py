import json
import os
import time
import requests
import markdown
from bs4 import BeautifulSoup
import base64
from typing import List
import re

from functools import lru_cache
from tqdm import tqdm 

# read github token at
GITHUB_TOKEN = json.load(open(os.path.expanduser('~/secret_key/github_tokens.json'), 'r'))['github']

#  ========================================================= github tokens
def get_readme_file_content(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        'Authorization': f'token {GITHUB_TOKEN}'
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        readme_info = response.json()
        content = base64.b64decode(readme_info['content']).decode('utf-8')
        return content
    else:
        print(f"Failed to fetch README file: {response.status_code}")
        return ''

def extract_titles_from_markdown(markdown_content: str) -> List[str]:
    html_content = markdown.markdown(markdown_content)
    soup = BeautifulSoup(html_content, 'html.parser')
    titles = []
    for i in range(1, 7):
        for header in soup.find_all(f'h{i}'):
            titles.append(header.get_text())
    return titles

def wash_title_tokens(titles: List[str]) -> List[str]:
    processed_tokens = []
    for title in titles:
        tokens = title.split()
        for token in tokens:
            cleaned_token = re.sub(r'[^a-zA-Z0-9]', '', token).lower()
            processed_tokens.append(cleaned_token)
    return list(set(processed_tokens))


def test_readme_title():
    test_cases = [
        ['roundcube', 'roundcubemail'],
        ['curl', 'curl'],
        ['libcpr', 'cpr']
    ]
    for owner, repo in test_cases:
        print(f"Project: {owner}/{repo}")
        readme_content = get_readme_file_content(owner, repo)
        titles = extract_titles_from_markdown(readme_content)
        print(*titles, sep='\n')
        print('=========')
        
# use cache to store all (owner, repo) -> stop tokens
# thus minimize the number of requests to github
@lru_cache(maxsize=1000000)
def get_github_readme_title_tokens(owner: str, repo: str) -> List[str]:
    '''get github readme title tokens and owner, repo
    stop tokens
    '''
    if os.path.exists('project_2_stoptokens.json'):
        with open('project_2_stoptokens.json', 'r') as fp:
            project_2_stoptokens = json.load(fp)
        if f'{owner}/{repo}' in project_2_stoptokens:
            return project_2_stoptokens[f'{owner}/{repo}']
    res = wash_title_tokens(extract_titles_from_markdown(get_readme_file_content(owner, repo))) + [owner, repo]
    time.sleep(1)  # limit to 10 per seconds
    return res

#  ============================= preprocess data
def load_cve_2_porject_info(data_dir: str):
    '''all necessary data for function sort_token'''
    output_file = os.path.join(data_dir, 'cve_2_porject_info.json')
    if not os.path.exists(output_file):
        cve_2_porjecte = dict()
        for fold in ["train", "test", "valid"]:
            fin = open(os.path.join(data_dir, f"tf_idf_filtered_bert_input_{fold}.json"), "r")
            for line in tqdm(fin, desc=f'Augement github_title_tokens and stored to ./cve_2_porject_info.json for {fold}'):
                data = json.loads(line)
                if data['cve'] not in cve_2_porjecte:
                    owner, repo = data['owner_repo'].split('/')
                    cve_2_porjecte[data['cve']] = {'cve': data['cve'], #'text': data['text'],
                                                    'owner_repo': data['owner_repo'], 'commit': data['commit'],
                                                    'label': data['label'],
                                                    'github_title_tokens': get_github_readme_title_tokens(owner, repo)
                                                    }
                # else:
                #     owner, repo = data['owner_repo'].split('/')
                #     cve_2_porjecte[data['cve']].append({'cve': data['cve'], # 'text': data['text'],
                #                                         'owner_repo': data['owner_repo'], 'commit': data['commit'],
                #                                         'label': data['label'],
                #                                         'github_title_tokens': get_github_readme_title_tokens(owner, repo)
                #                                         })
        with open(output_file, 'w') as fp:
            json.dump(cve_2_porjecte, fp)
    else:
        with open(output_file, 'r') as fp:
            cve_2_porjecte = json.load(fp)
    return cve_2_porjecte

def get_project_2_stoptokens_from_cve2project_info():
    '''get project_2_stoptokens from cve_2_porject_info'''
    cve_2_porjecte = load_cve_2_porject_info()
    project_2_stoptokens = dict()
    for cve, projects in cve_2_porjecte.items():
        for project in projects:
            if project['owner_repo'] not in project_2_stoptokens:
                project_2_stoptokens[project['owner_repo']] = project['github_title_tokens']
    return project_2_stoptokens

if __name__ == '__main__':
    # test_readme_title()
    project_2_stoptokens = get_project_2_stoptokens_from_cve2project_info()
    with open('project_2_stoptokens.json', 'r') as fp:
        project_2_stoptokens = json.load(fp)
    
    print(f'length of project_2_stoptokens: {len(project_2_stoptokens)}')
