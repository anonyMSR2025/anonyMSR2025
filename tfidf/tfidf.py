import numpy as np
from typing import Generator, List, Tuple, Dict




# ======================================================== similarity  ===============================================

def generate_inverted_index(data: List[Dict]) -> Dict:
    '''
    生成 word -> doc_index 的映射, 
    用来快速过滤掉和 query 不相关的 doc
    '''
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
    '''
    输入 的 query 和 所有 doc 的 embedding， 都是 word -> tfidf 的 dict
    输出 所有 doc 和 query 的相似度
    '''
    length = len(embeddings)
    inv_idx_dict = generate_inverted_index(embeddings)
    to_compare_indexes = []
    # find all the document indexes that have a common word with the current doc
    # logger.debug(f'Query keys: {query.keys()}', )
    # logger.debug(f'inv_idx_dict keys: {inv_idx_dict.keys()}')

    for word in query.keys():
        if word in inv_idx_dict: # inverted index does not incorporate query
            to_compare_indexes.extend(inv_idx_dict[word])

    # eliminate duplicates
    to_compare_indexes = list(set(to_compare_indexes))

    # calculate the similarity onlf if the id is larger than
    # the current document id for better efficiency
    similarities = np.zeros(len(embeddings))
    cur_doc_sims = []
    # if length > 3000:
    #     # pybar = tqdm(total=len(to_compare_indexes), desc='Case candidate > 5000:')
    #     start = time.time()
        
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
    # if length > 3000:
    #     print(f'total candidates: {length}, filtered candidates: {len(to_compare_indexes)}, time use: {(time.time() - start)/60: .2f} min')
    return similarities
