# input dataset
1. cve & patch: processed_data/clone_and_extraction_task_list.csv
2. mitre, cve reserve data: "mitre" dataset
3. cve description: tfidf/data/cve2desc.json, exported by load_cve2_desc.py from "NVD" dataset


# pipeline
## 1. 下载 commits & diffs
python clone_and_extract.py --task_fpath processed_data/clone_and_extraction_task_list.csv

输出结果在 data/commits, data/diff 文件夹内，格式 head -n 30 一个 sample 查看

## 2. timewindow filter & 生成 cand2date.json & 生成 valid_list.csv
python prepare_data.py --windowsize 400 --task_fpath processed_data/clone_and_extraction_task_list.csv --mitre_rootdir ../mitre/mitre

需要依赖 mitre/mitre/cvelistV5-main/cves 数据输入
processed_data/valid_list.csv: clone_and_extraction_task_list.csv里面valid的repo，可以用valid_list.csv看一下用window filter的recall
cve2cand_date.json: 每个valid的cve的起止时间
tfidf_data/inrepo_idf: repo内的idf


## 3. 计算tfidf
## 3.1 计算 df
python compute_df.py 

结果存储在 data/df

## 3.2 计算 idf
python merge_df.py

根据 data/df 中的结果计算
结果存储在 tfidf_data/idf_diff.json 和 tfidf_data/idf_commit.json
inrepo idf 的结果存储在 tfidf_data/inrepo_idf

## 3.3 计算tfidf
python compute_tfidf.py 

结果在 data/tfidf，计算后的 patch rank 在 tfidf_data/tfidf_result.csv

## 3.4 计算 tfidf 和 timwindow 合并的 rank 结果
```
python combine_ranker.py --log_dis
```


依赖于 prepare_data.py 输出的timewindow的详细中间结果 tfidf_data/commit_distance_details/*
    和 compute_tfidf.py 输出的详细中间结果 tfidf_data/tfidf_detail/*
输出结果 combine_ranker_res.csv

# 计算结果
```
python3 analysis_res.py
```

计算 timewindow 和 tfidf 的 recall


## 4. 导出 top 100，得到 MSR 的输入格式
```
python3 export_top_100.py
```

依赖 combine_ranker.py 输出的 combine_ranker_top_100.csv