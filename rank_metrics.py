import os
import numpy as np
# import ireval
import json
from typing import List, Dict
from tqdm import tqdm
import time
import logging
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, confusion_matrix, matthews_corrcoef, accuracy_score, ndcg_score
from itertools import groupby
from operator import itemgetter
from typing import Iterable, Tuple, Callable, List, ClassVar, Dict, Any, Union
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("debug.log", mode='a', delay=False),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def cosine_similarity(query, embeddings) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # deal wiht 0
    embeddings_normalized = embeddings / norms
    query = query / np.linalg.norm(query)
    # commit_message_embed = commit_message_embed / np.linalg.norm(commit_message_embed)
    # sim 
    cos_similarities = np.dot(embeddings_normalized, query.T)
    return cos_similarities


def generate_inverted_index(data: List[Dict]) -> Dict:
    inv_idx_dict = {}
    for index, doc_dict in enumerate(data):
        for word, tfidf in doc_dict.items():
            if word not in inv_idx_dict.keys():
                inv_idx_dict[word] = [index]
            elif index not in inv_idx_dict[word]:
                inv_idx_dict[word].append(index)
    return inv_idx_dict

def find_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u == 0 or norm_v == 0 or np.isnan(norm_u) or np.isnan(norm_v):
        return 0
    return np.dot(u, v) / (norm_u * norm_v)


def cosine_similarity_inverted_indenx(query: Dict, embeddings: List[Dict], normalize: bool=True) -> np.ndarray:
    length = len(embeddings)
    inv_idx_dict = generate_inverted_index(embeddings)
    to_compare_indexes = []
    # find all the document indexes that have a common word with the current doc
    logger.debug(f'Query keys: {query.keys()}', )
    logger.debug(f'inv_idx_dict keys: {inv_idx_dict.keys()}')
    for word in query.keys():
        if word in inv_idx_dict: # inverted index does not incorporate query
            to_compare_indexes.extend(inv_idx_dict[word])

    # eliminate duplicates
    to_compare_indexes = list(set(to_compare_indexes))

    # calculate the similarity onlf if the id is larger than
    # the current document id for better efficiency
    similarities = np.zeros(len(embeddings))
    cur_doc_sims = []
    if length > 3000:
        # pybar = tqdm(total=len(to_compare_indexes), desc='Case candidate > 5000:')
        start = time.time()
    for compare_doc_index in to_compare_indexes:
        embed = embeddings[compare_doc_index]
        word_set = list(set(list(query.keys()) + list(embed.keys())))
        word2idx = {word: idx for idx, word in enumerate(word_set)}
        qury_embed = np.zeros(len(word_set))
        candidate_embed = np.zeros(len(word_set))
        for w, tfidf in query.items():
            qury_embed[word2idx[w]] = tfidf
        for w, tfidf in embed.items():
            candidate_embed[word2idx[w]] = tfidf
        if normalize:
            sim = find_similarity(qury_embed, candidate_embed)
        else:
            sim = np.dot(qury_embed, candidate_embed)
        similarities[compare_doc_index] = sim
        # if length > 5000:
            # pybar.update(1)
        # cur_doc_sims.append([compare_doc_index, sim])
    if length > 3000:
        print(f'total candidates: {length}, filtered candidates: {len(to_compare_indexes)}, time use: {(time.time() - start)/60: .2f} min')
    return similarities

def get_ap_1(scores, labels):
    '''
    AP = \frac{1}{R} \sum{R}{k=1} num_{correct}(k) / k
    '''
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]
    ap = 0
    num_relevant = np.sum(labels)
    num_retrieved = 0
    num_relevant_retrieved = 0
    for i in range(len(scores)):
        if sorted_labels[i] == 1:
            num_relevant_retrieved += 1
            ap += num_relevant_retrieved / (i + 1)
    return ap / num_relevant


def get_ap_2(scores, labels):
    '''
    AP = \frac{1}{R} \sum{R}{k=1} P@k
    '''
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]
    
    precision_at_k = np.zeros(len(scores))
    num_relevant = np.sum(labels)
    num_retrieved = 0
    num_relevant_retrieved = 0
    for i in range(len(scores)):
        if sorted_labels[i] == 1:
            num_relevant_retrieved += 1
        num_retrieved += 1
        precision_at_k[i] = num_relevant_retrieved / num_retrieved
        
    return np.mean(precision_at_k)


def get_precision_recall_at_k(scores, labels, k=5):
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]
    origin_k = k
    k = min(k, len(scores))
    precision_at_k = np.zeros(k)
    recall_at_k = np.zeros(k)
    num_relevant = np.sum(labels)
    num_retrieved = 0
    num_relevant_retrieved = 0
    for i in range(k):
        if sorted_labels[i] == 1:
            num_relevant_retrieved += 1
        num_retrieved += 1
        precision_at_k[i] = num_relevant_retrieved / num_retrieved
        if num_relevant == 0:
            recall_at_k[i] = 0
        else:
            recall_at_k[i] = num_relevant_retrieved / num_relevant
        
    # padding results
    k = origin_k
    if len(precision_at_k) < k:  
        # print(f"padding precision_at_k with {k - len(precision_at_k)}")
        last_precision_value = precision_at_k[-1] if len(precision_at_k) > 0 else 0
        last_recall_value = recall_at_k[-1] if len(recall_at_k) > 0 else 0
        precision_at_k = np.pad(precision_at_k, (0, max(0, k - len(precision_at_k))), 'constant', constant_values=last_precision_value)
        recall_at_k = np.pad(recall_at_k, (0, max(0, k - len(recall_at_k))), 'constant', constant_values=last_recall_value)


    return precision_at_k, recall_at_k



# metrics
def sigmoid(z: np.ndarray):
    return 1 / (1 + np.exp(-z))

def np_divide(a, b):
    # return 0 with divide by 0 while performing element wise divide for np.arrays
    return np.divide(a, b, out=np.zeros_like(a), where=(b != 0))

def f1_score(p, r):
    if p + r < 1e-5:
        return 0.0
    return 2 * p * r / (p + r)

def modified_topk(predictions, labels, k):
    # print(predictions, labels)
    rerank = [item for item in zip(predictions, labels)]
    rerank.sort(reverse=True)

    if sum(labels) == 0:
        return None
    hits = sum([item[1] for item in rerank[:k]])
    prec = hits / min(k, sum(labels))
    rec = hits / sum(labels)
#     print(prec, rec)
    return prec, rec, f1_score(prec, rec)

def compute_metrics(scores, labels, topk, cves: List[str]=None) -> Dict:
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
    cves = [cve.split('/')[0] for cve in cves]  # format: cve/commit_hash
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    print(f'sample of scores: {scores[:10]}')
    print(f'sample of labels: {labels[:10]}')
    print(f'sample of cves: {cves[:10]}')
    # while computing f1 = (2 * precision * recall) / (precision + recall), some element in (precision+recall) will be 0
    f1 = np_divide(2 * precision * recall, precision + recall)

    f1_idx = np.argmax(f1)
    f1_best = f1[f1_idx]

    # mcc = np.array([matthews_corrcoef(labels, (scores >= threshold).astype(int)) for threshold in thresholds])
    # mcc_idx = np.argmax(mcc)
    # mcc_best = mcc[mcc_idx]

    # acc = np.array([accuracy_score(labels, (scores >= threshold).astype(int)) for threshold in thresholds])
    # acc_idx = np.argmax(acc)
    # acc_best = acc[acc_idx]

    auc = roc_auc_score(y_true=labels, y_score=scores)
    pred = [1 if scores[x] > 0.5 else 0 for x in range(len(scores))]

#    # confusion matrix
#     predictions = (scores >= thresholds[f1_idx]).astype(int)
#     cm = confusion_matrix(labels, predictions)
#     cm_list = cm.tolist()

    # precision recall at k
    # print(scores)
    # scores = list(scores)
    # print(scores)
    # raise Exception('stop here')
    if cves is not None:
        combined = list(zip(scores, labels, cves))
        combined.sort(key=itemgetter(2))
        precision_at_k_l, recall_at_k_l, ap_scikit_l, count_l, ndcg_l, ap_2_l = [], [], [], [], [], []
        for cve, group in groupby(combined, key=itemgetter(2)):
            #print(cve)
            cur_scores = []
            cr_labels = []
            for score, label, _ in group:
                cur_scores.append(score)
                cr_labels.append(label)
            precision_at_k, recall_at_k = get_precision_recall_at_k(np.array(cur_scores).reshape(-1,), np.array(cr_labels), k=topk)
            # print(cve)
            # print(np.array(cur_scores).reshape(-1,))
            # print(np.array(cr_labels))
            # print(f'precision_at_k: {precision_at_k}\n recall_at_k: {recall_at_k}')
            precision_at_k_l.append(precision_at_k)
            recall_at_k_l.append(recall_at_k)     

            # Calculate AP for the current group
            if np.sum(cr_labels) > 0:
                ap = average_precision_score(np.array(cr_labels), np.array(cur_scores).reshape(-1,))
                ap_scikit_l.append(ap)
                #ndcg_l.append(ndcg_score(np.array([cr_labels]), np.array([cur_scores])))
                # ap_1_l.append(get_ap_1(np.array(cur_scores).reshape(-1,), np.array(cr_labels)))
                # ap_2_l.append(get_ap_2(np.array(cur_scores).reshape(-1,), np.array(cr_labels)))
            else:
                ap = 0
                
            count_l.append(len(cr_labels))

        total_precision_at_k_array = np.stack(precision_at_k_l)
        precision_at_k = np.nanmean(total_precision_at_k_array, axis=0)
        total_recall_at_k_array = np.stack(recall_at_k_l)
        recall_at_k = np.nanmean(total_recall_at_k_array, axis=0)  
        
        # Calculate MAP
        map_score_scikit = np.mean(ap_scikit_l)
        #ndcg = np.mean(ndcg_l)
        # map_score_1 = np.mean(ap_1_l)
        # map_score_2 = np.mean(ap_2_l)
    else:
        precision_at_k, recall_at_k, map_score_scikit = 'N/A', 'N/A', 'N/A'

    return {"auc": auc,
            # "accuracy_best": acc_best,
            # "accuracy_threshold": thresholds[acc_idx],
            "precision": precision_score(labels, pred),
            "recall": recall_score(labels, pred),
            "f1": f1_score(labels, pred),
            "accuracy": accuracy_score(labels, pred),
            "f1_best": f1_best,
            "f1_threshold": thresholds[f1_idx],
            # 'confusion_matrix': cm_list,
            "precision@k": list(precision_at_k),
            "recall@k": list(recall_at_k),
            "precision@kl": precision_at_k_l,
             'MAP': map_score_scikit,
            #"NDCG": ndcg,
            # 'MAP_1': map_score_1,
            # 'MAP_2': map_score_2,
            'count': sum(count_l)/len(count_l),
            # "precision_1": sum(ps) / len(ps),
            # "recall_1": sum(rs) / len(rs),
            # "f1_1": f1_score(sum(ps) / len(ps), sum(rs) / len(rs)),
            # "mcc_best": mcc_best,
            # "mcc_threshold": thresholds[mcc_idx],
            "num_samples": len(labels)}

def precision_recall_f1(preds, labels):
    preds = preds.reshape(-1).tolist()
    labels = labels.reshape(-1).tolist()
    # print('inputed preds', preds[:10])
    # print('inputed labels', labels[:10])
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        else:
            tn += 1
    print("Confusion Matrix:")
    print(f"TP (True Positive): {tp}")
    print(f"FP (False Positive): {fp}")
    print(f"FN (False Negative): {fn}")
    print(f"TN (True Negative): {tn}")  # 假设 TN 为 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def plot_metrics(valid_metrics, test_metrics, title, metric_name='Metric'):
    """
    Plot the metrics for validation and test sets over epochs.

    Parameters:
    valid_metrics (list): List of validation metrics for each epoch.
    test_metrics (list): List of test metrics for each epoch.
    metric_name (str): Name of the metric, used for the title and labels of the plot.
    """
    epochs = range(1, len(valid_metrics) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, valid_metrics, 'b-', label='Validation')
    plt.plot(epochs, test_metrics, 'r-', label='Test')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    
    plt.grid(True)
    # plt.show()
    if not os.path.exists('plots'):
        os.mkdir('plots')
    
    plt.savefig(f'plots/{title}.png')
    print(f'Plot saved as plots/{title}.png')
    plt.show()

if __name__ == '__main__':
    # test plot_metrics
    
    valid_metrics = [0.1, 0.4, 0.3, 0.4, 0.5]
    test_metrics = [0.15, 0.25, 0.3, 0.6, 0.55]
    title = 'Test_Plot'
    metric_name = 'Test Metric'
    #plot_metrics(valid_metrics, test_metrics, title, metric_name)
    
