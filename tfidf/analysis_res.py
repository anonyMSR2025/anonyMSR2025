
import json
import pandas as pd
from typing import Tuple
from utils.debug import *

def calculate_recall(diff_list, time_window: Tuple[int, int]):
    recall = 0
    for diff in diff_list:
        if diff > time_window[0] and diff <= time_window[1]:
            recall += 1
    return recall / len(diff_list)

def calculate_recall_by_cve_id(df, time_window: Tuple[int, int]):
    recall = []
    weight = []
    for cve, cve_df in df.groupby("cve_id"):
        recall.append(calculate_recall(cve_df["rank"].tolist(), time_window))
        weight.append(len(cve_df))
    recall_ = [(r * w) for r, w in zip(recall, weight)]
    recall_res = sum(recall_) / sum(weight)
    return recall_res

def main_time_window():
    df = pd.read_csv("processed_data/valid_list.csv", header=0)
    print(len(df))
    # import pdb; pdb.set_trace()
    reserve_diff_commit_l = df["diff_idx"].tolist()
    # publish_diff_commit_l = df["diff_idx"].tolist()
    print(f'min reserve diff commit: {min(reserve_diff_commit_l)}, max reserve diff commit: {max(reserve_diff_commit_l)}')
    # print(f'min publish diff commit: {min(publish_diff_commit_l)}, max publish diff commit: {max(publish_diff_commit_l)}')


    for window in [(-25, 25), (-50, 50), (-100, 100), (-150, 150), (-200, 200), (-300, 300), (-400, 400), (-500, 500)]:
        recall_reserve = calculate_recall(reserve_diff_commit_l, window)
        # recall_publish = calculate_recall(publish_diff_commit_l, window)
        print(f"Window: {window}, Recall: {recall_reserve}")

def main_tfidf(fpath, method="tfidf"):
    # res_df = pd.read_csv("tfidf_data/tfidf_result.csv", header=0)
    res_df = pd.read_csv(fpath, header=0)

    # res_df2 = pd.read_csv("voyage_output_2.csv", header=0)
    # res_df = pd.concat([res_df, res_df2], ignore_index=True, axis=0)
    # import pdb; pdb.set_trace()
    # set columns 
    res_df.columns = ["cve_id","repo","owner","patch","rank"]
    #  NOTE: -1 表示这个 patch commit 没有被 time window filter 保留下来
    # 一种情况是，没有 recall 的 commit 是 time_window 2-3% 没有 recall 的 commit
    # 另一种情况是，这个 commit 没有出现在 commit history 中，在 analysis_alldb.py: line 102 被过滤掉了
    print('count of cves before filtering', len(res_df["cve_id"].unique().tolist()))
    res_df = res_df[res_df["rank"] != -1]
    cves = res_df["cve_id"].unique().tolist()
    print('count of cves after filtering', len(cves))
    # import pickle
    # valid_cve_set = pickle.load(open('./data/total_valid_cve_set.pkl', 'rb'))
    print(len(res_df))
    # res_df = res_df[res_df["cve_id"].isin(valid_cve_set)]
    # print(len(res_df))
    # plot tfidf recalled index 的分布
    # plot_histogram(res_df["rank"].tolist(), bins=100, title="TFIDF rank patch Index", xlabel="Index", savefilename="tfidf_ranked_gt_index.png")

    for window in [(0, 10), (0, 50), (0, 80), (0, 100), (0, 120), (0, 150), (0, 200), (0, 250), (0, 300), (0, 400), (0, 600), (0, 800)]:
        recall_tfidf = calculate_recall(res_df["rank"].tolist(), window)
        recall_tfidf_by_cve_id = calculate_recall_by_cve_id(res_df, window)
        print(f"{method} Recall@{window[1]} = {recall_tfidf:.4f}, {method} Recall@{window[1]} by cve_id = {recall_tfidf_by_cve_id:.4f}")

if __name__ == "__main__":
    print("time window")
    main_time_window()
    print("tfidf")
    main_tfidf("tfidf_data/tfidf_res.csv", method="tfidf")
    # main_tfidf("codesage_output.csv", method="codesage")
    main_tfidf("combine_ranker_res.csv", method="combine_ranker")
    # main_tfidf("voyage_output.csv", method="voyage")
    # print("voyage")
    # main_tfidf("voyage_output.csv")