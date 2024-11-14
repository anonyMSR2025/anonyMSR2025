import json
import re
import numpy as np
from numpy import array, exp
import pandas as pd
from typing import Dict, List 

### this file contain different attention mask calculation from the n masks from n annotators. In this code there are 3 annotators


#### Few helper functions to convert attention vectors in 0 to 1 scale. While softmax converts all the values such that their sum lies between 0 --> 1. Sigmoid converts each value in the vector in the range 0 -> 1.

##### We mostly use softmax
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
 
def neg_softmax(x):
    """Compute softmax values for each sets of scores in x. Here we convert the exponentials to 1/exponentials"""
    e_x = np.exp(-(x - np.max(x)))
    return e_x / e_x.sum(axis=0)
def sigmoid(z):
      """Compute sigmoid values"""
      g = 1 / (1 + exp(-z))
      return g


##### This function is used to aggregate the attentions vectors. This has a lot of options refer to the parameters explanation for understanding each parameter.
def aggregate_attention(at_mask, params):
    """input: attention vectors from 2/3 annotators (at_mask), row(dataframe row), params(parameters_dict)
       function: aggregate attention from different annotators.
       output: aggregated attention vector"""
       
    #### If the final label is normal or non-toxic then each value is represented by 1/len(sentences)
    # if row['final_label'] in ['normal', 'non-toxic']:
    #     at_mask_fin = [1/len(at_mask[0]) for x in at_mask[0]]
    # else:
    at_mask_fin = at_mask
    #### Else it will choose one of the options, where variance is added, mean is calculated, finally the vector is normalised.   
    if (params['type_attention'] == 'sigmoid'):
        at_mask_fin = int(params['variance']) * at_mask_fin
        at_mask_fin = np.mean(at_mask_fin,axis=0)
        at_mask_fin = sigmoid(at_mask_fin)
    elif (params['type_attention'] == 'softmax'):
        at_mask_fin = int(params['variance']) * at_mask_fin
        at_mask_fin = np.mean(at_mask_fin, axis=0)
        at_mask_fin = softmax(at_mask_fin)
    elif (params['type_attention']=='neg_softmax'):
        at_mask_fin = int(params['variance'])*at_mask_fin
        at_mask_fin = np.mean(at_mask_fin,axis=0)
        at_mask_fin = neg_softmax(at_mask_fin)
    elif (params['type_attention'] in ['raw', 'individual']):
        pass
    
    if (params['decay']==True):
         at_mask_fin = decay(at_mask_fin,params)

    return at_mask_fin
    
    
    
##### Decay and distribution functions. To decay the attentions left and right of the attented word. This is done to decentralise the attention to a single word. 
def distribute(old_distribution, new_distribution, index, left, right,params):
    window = params['window']
    alpha = params['alpha']
    p_value = params['p_value']
    method =params['method']
    reserve = alpha * old_distribution[index]
#     old_distribution[index] = old_distribution[index] - reserve
    
    if method=='additive':    
        for temp in range(index - left, index):
            new_distribution[temp] = new_distribution[temp] + reserve/(left+right)
        
        for temp in range(index + 1, index+right):
            new_distribution[temp] = new_distribution[temp] + reserve/(left+right)
    
    if method == 'geometric':
        # we first generate the geometric distribution for the left side
        temp_sum = 0.0
        newprob = []
        for temp in range(left):
            each_prob = p_value*((1.0-p_value)**temp)
            newprob.append(each_prob)
            temp_sum +=each_prob
            newprob = [each/temp_sum for each in newprob]
        
        for temp in range(index - left, index):
            new_distribution[temp] = new_distribution[temp] + reserve*newprob[-(temp-(index-left))-1]
        
        # do the same thing for right, but now the order is opposite
        temp_sum = 0.0
        newprob = []
        for temp in range(right):
            each_prob = p_value*((1.0-p_value)**temp)
            newprob.append(each_prob)
            temp_sum +=each_prob
            newprob = [each/temp_sum for each in newprob]
        for temp in range(index + 1, index+right):
            new_distribution[temp] = new_distribution[temp] + reserve*newprob[temp-(index + 1)]
    
    return new_distribution



def decay(old_distribution, params):
    window=params['window']
    new_distribution = [0.0] * len(old_distribution)
    for index in range(len(old_distribution)):
        right = min(window, len(old_distribution) - index)
        left = min(window, index)
        new_distribution = distribute(old_distribution, new_distribution, index, left, right, params)

    if (params['normalized']):
        norm_distribution = []
        for index in range(len(old_distribution)):
            norm_distribution.append(old_distribution[index] + new_distribution[index])
        tempsum = sum(norm_distribution)
        new_distribution = [each/tempsum for each in norm_distribution]
    return new_distribution




# get attention masked by tf-idf
# step 1, get tokens' indices
def load_cve_2_attention_tokens(threshold: float = 0.0, file_path :str = './cve_output_attention_tokens.xlsx'):
    '''
    '''
    data_df = pd.read_excel(file_path, engine='openpyxl')
    output_cve_2_attention_tokens = {}
    for idx, row in data_df.iterrows():
        # print(row)
        cve = row['cve']
        attention_tokens_raw = row['cve_output_tokens']
        commit = row['commit']
        commit_id = commit.split('commit/')[-1]
        attention_tokens = [item[0] for item in eval(attention_tokens_raw) if item[1] > threshold]
        output_cve_2_attention_tokens[(cve, commit_id)] = attention_tokens
    return output_cve_2_attention_tokens

def split_by_camel_snake_split_indices(sentence):
    
    split_pattern = r'(?<=[a-z])(?=[A-Z])|_|\s+|\.|\=|\,|\!|\?|\:|\;|\'|\"|\-|\(|\)|\{|\}|\[|\]|\>|\<|\/|\/|\#|\@|\%|\&|\*|\+|\~|\^|\`'
    split_pattern2 = r'_|\s+|\.|\=|\,|\!|\?|\:|\;|\'|\"|\-|\(|\)|\{|\}|\[|\]|\>|\<|\/|\/|\#|\@|\%|\&|\*|\+|\~|\^|\`'
    split_pattern3 = r'\s+|\=|\,|\!|\?|\:|\;|\'|\"|\(|\)|\{|\}|\[|\]|\>|\<|\/|\/|\#|\@|\%|\&|\*|\+|\~|\^|\`'

    def get_split_indices(pattern, text):
        return [match.start() for match in re.finditer(pattern, text)]

    indices1 = get_split_indices(split_pattern, sentence)
    indices2 = get_split_indices(split_pattern2, sentence)
    indices3 = get_split_indices(split_pattern3, sentence)
    return indices1, indices2, indices3

def is_subsequence(A: str, B: str) -> bool:
    '''
    字符串 A 是否可以通过删除字符串 B 中的某些字符得到 
    '''
    i, j = 0, 0
    while j < len(B):
        if i < len(A) and A[i] == B[j]:
            i += 1
        j += 1
    return i == len(A)

def highlight_attention_tokens(sentence: str, attention_tokens: dict) -> List[int]:
    token_indices = [0] * len(sentence)
    # get split indices
    for each_tuple_list_and_score in attention_tokens.values():
        each_tuple_list = each_tuple_list_and_score[0]
        each_tfidf_score = each_tuple_list_and_score[1]
        for each_tuple in each_tuple_list:
            for x in range(each_tuple[0], each_tuple[1]):
                if x < len(sentence):
                   token_indices[x] = each_tfidf_score
    return token_indices

def convert_cc(encoding, tokenizer, params):
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"]

    new_input_ids = []
    new_att_mask = []
    new_offset_mapping = []
    
    start_id = 0
    #if params["model_name"] == "microsoft/codebert-base":
    #    first_token = 50118
    #    second_token = 50118
    #    replace_token = 50140
    #else:
    #    first_token = 317
    #    second_token = 1022
    #    replace_token = 317
    while (start_id < len(input_ids)):
        if params["model_name"] == "microsoft/codebert-base" and input_ids[start_id] == 50118 and (start_id + 1 < len(input_ids) and input_ids[start_id + 1] == 50118):
            new_input_ids.append(50140)
            new_att_mask.append(attention_mask[start_id])
            new_offset_mapping.append(offset_mapping[start_id])
            start_id += 2
        else:
            new_input_ids.append(input_ids[start_id])
            new_att_mask.append(attention_mask[start_id])
            new_offset_mapping.append(offset_mapping[start_id])
            start_id += 1
    return {"input_ids": new_input_ids, "attention_mask": new_att_mask, "offset_mapping": new_offset_mapping}

# step 2, get attention mask
def preprocess_cc(text, tokenizer, params, is_code):
    if params["model_name"] == "microsoft/codebert-base":
        if is_code == True:
           text_ = text.replace("\n\n", "[NEWLINE]")
           text_ = text_.replace("\n", "[NEWLINE]")
           encoding_ = tokenizer.encode_plus(
               text_,
               return_offsets_mapping=True,
               add_special_tokens=True
           )

           encoding = convert_cc(tokenizer.encode_plus(
               text,
               return_offsets_mapping=True,
               add_special_tokens=True
           ), tokenizer, params)
           return encoding_, encoding, None
        else:
           text_ = re.sub("\n\xa0", " \xa0", text)
           text_ = text_.replace("\n\n", "[NEWLINE]")
           text_ = text_.replace("\n", "[NEWLINE]")
           encoding_ = tokenizer.encode_plus(text_, return_offsets_mapping=True, add_special_tokens=True)
           encoding = convert_cc(tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True), tokenizer, params)
        mapping = None
    elif params["model_name"] == "microsoft/unixcoder-base":
        text_, mapping = regex_sub_with_mapping("[ \t\n]{2,}", " ", text)
        encoding_ = tokenizer.encode_plus(text_, return_offsets_mapping=True, add_special_tokens=True)
        encoding = encoding_
    return encoding_, encoding, mapping

def regex_sub_with_mapping(pattern, replacement, original_string):
    # 初始化替换后到原始字符索引的映射字典
    reverse_mapping = {}
    new_string = ""
    current_new_index = 0

    last_end = 0  # 记录上一个匹配的结束位置

    # 遍历所有匹配项
    for match in re.finditer(pattern, original_string):
        start, end = match.span()

        # 复制并记录原始文本中未匹配的部分
        for i in range(last_end, start):
            reverse_mapping[current_new_index] = i
            new_string += original_string[i]
            current_new_index += 1

        # 添加替换后的内容
        for j in range(len(replacement)):
            reverse_mapping[current_new_index] = start
            new_string += replacement[j]
            current_new_index += 1

        last_end = end  # 更新未处理部分的起始位置

    # 处理匹配之后的剩余字符
    for i in range(last_end, len(original_string)):
        reverse_mapping[current_new_index] = i
        new_string += original_string[i]
        current_new_index += 1
    reverse_mapping[len(new_string)] = len(original_string)

    return new_string, reverse_mapping

def tokenize_with_gt_attention_mask(text: str, tokenizer, attention_tokens: dict, params=None, verbose=False, is_code=False, commit_id=0, part="cve"):
   # text = text.replace(' [SEP] ', '[SEP]').replace('\n', '').lower()
    if len(text) > 5120:  # max length of bert input text
        text = text[:5120]

    # if commit_id == "343c7bd381b63e042d437111718918f951d9b30d" and part == "commit":
    #     import pdb; pdb.set_trace()
        
    #attention_tokens = json.loads(attention_tokens)
    attention_index_mask = highlight_attention_tokens(text, attention_tokens)
    tokenizer.add_tokens(["[NEWLINE]"])

    encoding_, encoding, char_mapping = preprocess_cc(text, tokenizer, params, is_code) 
    assert len(encoding_["input_ids"]) == len(encoding["input_ids"])
    # print(encoding.keys())
    if verbose:
        print(tokens)
        
    input_ids = encoding_['input_ids']
    offsets = encoding['offset_mapping']
    gt_attention_mask = [0] * len(input_ids)
    for i, input_id, offset in zip(range(len(input_ids)), input_ids, offsets):
        if offset is None: continue
        start, end = offset 
        if params["model_name"] == "microsoft/unixcoder-base":
          start_before = char_mapping[start]
          end_before = char_mapping[end]
        elif params["model_name"] == "microsoft/codebert-base":
          start_before = start
          end_before = end
        if verbose:
            print(f"Token: {tokenizer.decode([input_id])}, start: {start}, end: {end}, token: {text[start: end]}")
        if start_before == end_before: continue
        this_slice = attention_index_mask[start_before: end_before]
        gt_attention_score = max(this_slice) if len(this_slice) > 0 else 0 # sum(attention_index_mask[start: end]) / (end - start)
        if gt_attention_score > 0:
            gt_attention_mask[i] = gt_attention_score
            
    # if verbose:
    #     for input_id, score in zip(input_ids, gt_attention_mask):
    #         if score > 0:
    #             print(f"Token: {tokenizer.decode([input_id])}, score: {score}")
    
    # process attention
    gt_attention_mask = np.array([gt_attention_mask])
    # print(gt_attention_mask.shape)
    # res = aggregate_attention(gt_attention_mask, params)
    # print(res)
    # raise Exception('stop')
    #import pdb; pdb.set_trace()
    #if params is None:
    return {'gt_attention': [float(item) for item in list(gt_attention_mask[0])], 'input_ids': input_ids, 'attention_mask': encoding_['attention_mask']}
    #else:
    #    return {'gt_attention': aggregate_attention(gt_attention_mask, params), 'input_ids': input_ids, 'attention_mask': encoding_['attention_mask']}
        
    
if __name__ == '__main__':
    with open('../bestModel_bert_base_uncased_Attn_train_TRUE.json', 'r') as fp:
        params = json.load(fp)
    
    sample_cve = 'CVE-2021-44273'
    sample_token = ['mitm', 'hostnames', 'transparent', 'servers', 'certificate', 'certificates']
    
    # with open('./cve_2_porject_info.json', 'r') as fp:
    #     cve_total_data = json.load(fp)
        
    # cve_data = cve_total_data[sample_cve]
    # print(len(cve_data))
    # print(cve_data[0].keys())
    # print(len(cve_data[0]['text']))
    
    text = '''e2guardian v5.4.x <= v5.4.3r is affected by missing ssl certificate validation in the ssl mitm engine. in standalone mode (i.e., acting as a proxy or a transparent proxy), with ssl mitm enabled, e2guardian, if built with openssl v1.1.x, did not validate hostnames in certificates of the web servers that it connected to, and thus was itself vulnerable to mitm attacks.[sep]fix spelling of 'implement' (and derived words).[sep]diff --git a/configs/e2guardian.conf.in b/configs/e2guardian.conf.inindex 062c0a82..2fb08c8d 100644--- a/configs/e2guardian.conf.in+++ b/configs/e2guardian.conf.in@@ -735,7 +735,7 @@ reverseaddresslookups = off # and normally e2g will no longer check site lists for ip's # if you want to keep backward list compatablity then set this to # 'on' - but note this incurs an overhead - putting ip in ipsitelists-# and setting this to off gives the fastest implimentation.+# and setting this to off gives the fastest implementation. # default is 'on'  ###@@ -1140,7 +1140,7 @@ weightedphrasemode = 2 ## things that will only work if specifically compliled  ## 'new' debug system (generaly compliled in release systems)-## note that this is only partialy implimented and only works for icap, clamav +## note that this is only partialy implemented and only works for icap, clamav  ## and icapc ## and so 'all' = 'icap,clamav,icapc' only. ## to debug other areas of code re-compiling in debug mode is required'''
    # necessary, other wise bert t
    # text = text.replace(' [SEP] ', '[SEP]').replace('\n', '').lower()
    print(text)
    
    from transformers import BertTokenizer, BertTokenizerFast, RobertaTokenizerFast
    # # tokenize_with_token_idx(text.replace('\n', ''), tokenizer)
    # tokenize_with_token_idx(text, tokenizer)
    # highLight_mark = highlight_attention_tokens(cve_data[0]['text'][0], sample_token)
    tokenizer = RobertaTokenizerFast.from_pretrained('microsoft/codebert-base')
    tokenize_with_gt_attention_mask(text, tokenizer, sample_token, params=params, verbose=True)
    # attention_token_index_match = highlight_attention_tokens(text, sample_token)
    # print(attention_token_index_match)
    # print()
    
    #
    
    # start_idxs = []
    # end_idxs = []
    # for i in range(len(highLight_mark) - 1):
    #     if highLight_mark[i] == 0 and highLight_mark[i+1] == 1:
    #         start_idxs.append(i)
    #     if highLight_mark[i] == 1 and highLight_mark[i+1] == 0:
    #         end_idxs.append(i)
    
    # assert len(start_idxs) == len(end_idxs)
    
    # for s, e in zip(start_idxs, end_idxs):
    #     print(cve_data[0]['text'][0][s-15:e+15])
    # print('attention token: ', sample_token)
