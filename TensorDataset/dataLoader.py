import torch
import transformers
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder



def custom_att_masks(text_ids):
    attention_masks = []

    # For each sentence...
    for sent in text_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return attention_masks

def custom_code_masks(code_ids):
    attention_masks = []

    # For each sentence...
    for code_list in code_ids:
        this_attention_masks = []
        for sent in code_list:

            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id > 0) for token_id in sent]   
            this_attention_masks.append(att_mask)

        # Store the attention mask for this sentence.
        attention_masks.append(this_attention_masks)
    return attention_masks

def combine_features(tuple_data, params, post_ids=None, is_train=False, batch_size=None):
    #print(f'sample of tuple_data: {tuple_data[0]}')
    #print(f'length of tuple_data: {len(tuple_data)}')
    # raise Exception('stop to check data input, attention_pred_mask added')
    sorted_ids = [ele[1] for ele in tuple_data]
    text_ids =  [ele[2] for ele in tuple_data]
    text_atts = [ele[3] for ele in tuple_data]
    text_masks = [ele[4] for ele in tuple_data] # this mask is not used at all, so we can pass an empty list before removing it and testing it 
    code_ids =  [ele[5] for ele in tuple_data]
    code_atts = [ele[6] for ele in tuple_data]
    code_masks = [ele[7] for ele in tuple_data]

    labels = [ele [0] for ele in tuple_data]
    # post_ids iterable 
    # print(f'sample of input_ids: {input_ids[0]}')
    # print(f'sample of att_vals: {att_vals[0]}')
    # raise Exception('stop to check data input')
    if post_ids is None:
        post_ids = [""] * len(text_ids)  # if post_ids is not provided, fill with empty strings, dummy values
        
    encoder = LabelEncoder()
    
    encoder.classes_ = np.load(params['class_names'],allow_pickle=True)
    labels=encoder.transform(labels)
    
    #print(f'sample of encoded labels: {labels[0]}')
    #print(f'positive ratio of labels: {np.sum(labels)/len(labels)}')

    for x in range(len(text_atts)):
        this_label = labels[x]
        #if this_label == 0:
        #    text_atts[x] = [0] * len(text_atts[x])

    text_ids = pad_sequences(text_ids,maxlen=int(params['max_length']), dtype="long", 
                          value=0, truncating="post", padding="post")
    text_atts = pad_sequences(text_atts,maxlen=int(params['max_length']), dtype="float", 
                          value=0.0, truncating="post", padding="post")
    text_masks = custom_att_masks(text_ids)

    new_code_ids = []
    new_code_atts = []
    code_max_num = params['codetopk']

    for x in range(len(code_ids)):
        len_x = len(code_ids[x])
        this_label = labels[x]
        #for y in range(len_x, 20):
        #    code_ids[x].append([])
        #    code_atts[x].append([])

        this_sorted_ids = sorted_ids[x][:code_max_num]
        this_code_ids = [code_ids[x][y] for y in this_sorted_ids]

        #if this_label == 1:
        this_code_atts = [code_atts[x][y] for y in this_sorted_ids]
        #else:
        #    this_code_atts = [[0] * len(code_atts[x][y]) for y in this_sorted_ids]

        new_code_ids.append(pad_sequences(this_code_ids,maxlen=int(params['max_length']), dtype="long", 
                            value=0, truncating="post", padding="post"))
        new_code_atts.append(pad_sequences(this_code_atts,maxlen=int(params['max_length']), dtype="float", 
                          value=0.0, truncating="post", padding="post"))
    code_masks = np.array(custom_code_masks(new_code_ids))
    #code_ids = np.array(new_code_ids)
    #code_atts = np.array(new_code_atts)
    
    dataloader = return_dataloader_trace(post_ids, params, labels, text_ids, text_atts, text_masks, new_code_ids, new_code_atts, code_masks, is_train, batch_size)
    return dataloader

def collate_fn(batch):
    return tuple(zip(*batch))

class CustomDataset(Dataset):
    def __init__(self, labels, post_ids, text_ids, text_atts, text_masks, code_ids, code_atts, code_masks):
        self.text_ids = text_ids
        self.text_atts = text_atts
        self.text_masks = text_masks
        
        self.code_ids = code_ids
        self.code_atts = code_atts
        self.code_masks = code_masks

        self.labels = labels
        self.strings = post_ids

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        label = self.labels[idx]
        string = self.strings[idx]

        text_id = self.text_ids[idx]
        text_att = self.text_atts[idx]
        text_mask = self.text_masks[idx]

        code_id = self.code_ids[idx]
        code_att = self.code_atts[idx]
        code_mask = self.code_masks[idx]

        return label, string, text_id, text_att, text_mask, code_id, code_att, code_mask

def return_dataloader_trace(post_ids, params, labels, text_ids, text_atts, text_masks, code_ids, code_atts, code_masks, is_train=False, batch_size=None):
    labels = torch.tensor(labels, dtype=torch.long)
    text_ids = torch.tensor(text_ids)
    text_masks = torch.tensor(np.array(text_masks), dtype=torch.uint8)
    text_atts = torch.tensor(np.array(text_atts), dtype=torch.float)

    code_ids = [torch.tensor(code_ids[x]) for x in range(len(code_ids))]
    code_masks = [torch.tensor(np.array(code_masks[x]), dtype=torch.uint8) for x in range(len(code_masks))]
    code_atts = [torch.tensor(np.array(code_atts[x]), dtype=torch.float) for x in range(len(code_atts))]

    data = CustomDataset(labels, post_ids, text_ids, text_atts, text_masks, code_ids, code_atts, code_masks)
    if (is_train==False):
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size or params["batch_size"], collate_fn = collate_fn)
    return dataloader


