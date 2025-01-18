import json
import pandas
import random
random.seed(42)

# 不能处理cve_desc_end_idx > 256的情况，要排除

path = "../../traceability/backup/Data/"

test_data = json.load(open(path + "Trace_microsoft-codebert-base/output_Dataset_test_ranked_attfirst.json", "r"))

with open(path + "tfidf_result_2.json", "r") as fin:
    data = json.load(fin)
    labels = data["labels"]
    ids = data["annotation_ids"]
    pred = data["preds"]
    #all_cves = set([ids[x].split("/")[0] for x in range(len(ids))])
    cve_to_idpredlabel = {}
    cve_to_pos_ids = {}
    for x in range(len(ids)):
        this_cve = ids[x].split("/")[0]
        cve_to_idpredlabel.setdefault(this_cve, [])
        if ids[x].startswith("CVE"):
           cve_to_idpredlabel[this_cve].append((ids[x], pred[x], labels[x]))
           if labels[x] == 1:
               cve_to_pos_ids.setdefault(this_cve, [])
               cve_to_pos_ids[this_cve].append(ids[x])
    cve_w_large_size = set([])
    for cve in cve_to_pos_ids.keys():
        this_cve_posids = cve_to_pos_ids[cve]
        assert len(this_cve_posids) == 1
        if this_cve_posids[0] and len(test_data[this_cve_posids[0]]["cve_desc_ids"]) >= 256: continue
        if len(cve_to_idpredlabel[cve]) > 5 and sum([x[2] for x in cve_to_idpredlabel[cve]]) > 0:
            cve_w_large_size.add(cve)
    print("cve w large size:", len(cve_w_large_size))
    selected_pos_postids = []
    selected_neg_postids = []
    NEG_COUNT = 3
    sampled_cve = sorted(random.sample(cve_w_large_size, 100))
    for each_cve in sampled_cve:
        idpredlabel_list = cve_to_idpredlabel[each_cve]
        sorted_list = sorted(idpredlabel_list, key = lambda x:x[1], reverse=True)
        this_pos_postid = None
        this_neg_postids = []
        for x in range(len(sorted_list)):
            #if this_pos_postid and this_neg_postid: break
            if sorted_list[x][2] == 1:
                this_pos_postid = sorted_list[x][0]
        for x in range(len(sorted_list)):
            if sorted_list[x][2] == 0:
                if len(test_data[sorted_list[x][0]]["cve_desc_ids"]) >= 256: continue
                this_neg_postids.append(sorted_list[x][0])
                if len(this_neg_postids) >= NEG_COUNT:
                    break
        #if (this_neg_postid is None) or (this_pos_postid is None):
        #    import pdb; pdb.set_trace()
        assert this_pos_postid and len(this_neg_postids) == NEG_COUNT
        selected_pos_postids.append(this_pos_postid)
        selected_neg_postids.append(this_neg_postids)
    postids = []
    labels = []
    for idx in range(100):
       this_postids = [selected_pos_postids[idx]] + selected_neg_postids[idx]
       this_labels = [1] + [0] * NEG_COUNT
       shuffled_id = [x for x in range(NEG_COUNT + 1)]
       random.shuffle(shuffled_id)
       postids += [this_postids[x] for x in shuffled_id]
       labels += [this_labels[x] for x in shuffled_id]
    output_df = pandas.DataFrame({"postids": postids, "labels": labels})
    output_df.to_csv("sampled_postids4.csv", sep=",")

