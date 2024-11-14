from transformers.models.roberta.modeling_roberta import *
from .utils import masked_cross_entropy
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F

class SC_weighted_BERT(RobertaPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.alpha = [1, 10]
        self.num_labels = config.num_labels
        self.pooling_method = params["pooling_method"]
        self.weights=params['weights']
        self.train_att= params['train_att']
        self.lam = params['att_lambda']
        self.num_sv_heads=params['num_supervised_heads']
        self.sv_layer = params['supervised_layer_pos']
        if params["model_name"] == "microsoft/codebert-base":
            self.bert = RobertaModel.from_pretrained(params["model_name"], config=config)
        elif params["model_name"] == "microsoft/unixcoder-base":  # unixcoder
            self.bert = RobertaModel.from_pretrained(params["model_name"], config=config)

        #    self.bert.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.kernel_sizes = [1, 2, 3]  # 可以根据需要选择合适的卷积核尺寸
        self.num_filters = 128         # 卷积核数量
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=config.hidden_size, out_channels=self.num_filters, kernel_size=k)
            for k in self.kernel_sizes
        ])

        self.classifier = nn.Linear(config.hidden_size + len(self.kernel_sizes) * self.num_filters, config.num_labels)

    def get_pooled_output(self, input_ids, mask):
        outputs = self.bert(input_ids,attention_mask= mask,
        )
        pooled_output = outputs[1]
        return pooled_output, outputs

    def get_upperlayer_output(self, text_output, code_output):
        concatenated_output = torch.concat([text_output, code_output], axis=1)
        pooled_output = self.dropout(concatenated_output)
        logits = self.classifier(pooled_output)
        return logits, F.softmax(logits, dim=1)

    # def get_avg_code_pool()
    
    def get_attention_loss(self, this_outputs, this_att, this_mask):
        loss_att= []
        for i in range(self.num_sv_heads):
            attention_weights = this_outputs[2][self.sv_layer][:,i,0,:]
            loss_att.append(masked_cross_entropy(attention_weights,this_att,this_mask))
        return torch.stack(loss_att)
 
    def forward(self,
        input_ids=None,
        text_att=None,
        text_mask=None,
        code_ids=None,
        code_att=None,
        code_mask=None,
        code_lengths=None,
        labels=None,
        device=None):

        code_ids_ = code_ids.data.to(device)
        code_mask_ = code_mask.data.to(device)
        code_att_ = code_att.data.to(device)

        extended_weight = []
        for i in range(len(code_lengths)):
            extended_weight.extend([1.0 / code_lengths[i]] * code_lengths[i]) 
        extended_weight_ = torch.tensor(extended_weight).to(device)

        text_pooled_outputs, text_outputs = self.get_pooled_output(input_ids, text_mask)
        code_pooled_outputs, code_outputs = self.get_pooled_output(code_ids_, code_mask_)

        pooling_method = "max"

        if self.pooling_method == "avg":
            weighted_code_outputs = code_pooled_outputs * extended_weight_.view(-1, 1)
            avg_code_outputs = []
            left_idx = 0
            for i in range(len(code_lengths)):
                right_idx = left_idx + code_lengths[i]
                avg_code_outputs.append(weighted_code_outputs[left_idx:right_idx, :].sum(dim=0))
                left_idx = right_idx
            assert right_idx == len(extended_weight_)
            avg_code_outputs = torch.stack(avg_code_outputs).to(device)
        elif self.pooling_method == "max":
            avg_code_outputs = []
            left_idx = 0
            for i in range(len(code_lengths)):
                right_idx = left_idx + code_lengths[i]
                avg_code_outputs.append(torch.max(code_pooled_outputs[left_idx:right_idx, :], dim=0)[0])
                left_idx = right_idx
            avg_code_outputs = torch.stack(avg_code_outputs).to(device)
        elif self.pooling_method == "cnn":
            avg_code_outputs = []
            left_idx = 0
            for i in range(len(code_lengths)):
                right_idx = left_idx + code_lengths[i]
                code_chunk = code_pooled_outputs[left_idx:right_idx, :]  # [seq_len, hidden_size]
                left_idx = right_idx
                # 添加 batch 维度并调整形状以适配 Conv1d
                code_chunk = code_chunk.unsqueeze(0).transpose(1, 2)  # [1, hidden_size, seq_len]
                conv_outputs = []
                for conv, kernel_size in zip(self.convs, self.kernel_sizes):
                    if code_chunk.shape[0] >= kernel_size:
                        conv_output = torch.relu(conv(code_chunk))  # [1, num_filters, seq_len_out]
                        # 全局最大池化
                        pooled_output = torch.max(conv_output, dim=2)[0]  # [1, num_filters]
                    else:
                        pooled_output = torch.zeros(1, self.num_filters).to(self.device)
                    conv_outputs.append(pooled_output)
                assert conv_outputs
                # 将不同卷积核尺寸的输出拼接
                total_pooled_output = torch.cat(conv_outputs, dim=1)  # [1, num_filters * len(kernel_sizes)]
                avg_code_outputs.append(total_pooled_output.squeeze(0)) 
            avg_code_outputs = torch.stack(avg_code_outputs).to(device)
        
        logits, proba = self.get_upperlayer_output(text_pooled_outputs, avg_code_outputs)

        outputs = (logits,) + text_outputs[2:] + code_outputs[2:]

        loss_funct = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device)) #, reduction='none')

        if labels is not None:

            loss_logits = loss_funct(logits.view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
            outputs = (loss,) + outputs
        
        return outputs
        
