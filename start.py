import os
import copy
import json
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from model.chineseBert import Bert
from pypinyin import pinyin, Style
from transformers import AutoModel, AutoTokenizer
import torch.nn.init as init

max_length = 128
batch_size = 32
epochs = 20

log_after_step = 20

model_path = './outputs/summaryBert/pretrained/'
os.makedirs(model_path, exist_ok=True)
model_path = model_path + 'v1.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
    
class CorrectionNetwork(nn.Module):

    def __init__(self):
        super(CorrectionNetwork, self).__init__()
        self.tokenizer = BertTokenizerFast('ChineseBERT-base/vocab.txt')
        self.bert = Bert.from_pretrained('ChineseBERT-base')
        self.word_embedding_table = self.bert.get_input_embeddings()
        self.dense_layer = nn.Linear(self.bert.config.hidden_size, len(self.tokenizer))

    def forward(self, sequences, detect_hidden_states):
        pinyin_ids = self.get_pinyin_ids(sequences).to(device)
        input_ids = self.tokenizer(sequences, padding='max_length', max_length=max_length, return_tensors='pt',
                                truncation=True)['input_ids'].to(device)
        bert_outputs = self.bert(input_ids, pinyin_ids)
        hidden_states = bert_outputs['last_hidden_state'] + detect_hidden_states
        correction_outputs = self.dense_layer(hidden_states)
        error_weight = self.cal_error_weight(correction_outputs.argmax(dim=-1), input_ids).to(device)
        return self.dense_layer(hidden_states) * error_weight.unsqueeze(1) # 算loss
    
    def get_pinyin_ids(self, sequences):
        pinyin_ids_arr = []
        for idx, sequence in enumerate(sequences):
            tokenizer_output = self.tokenizer(sequence, truncation=True, padding='max_length', max_length=128, return_offsets_mapping=True)
            pinyin_ids = self.convert_sentence_to_pinyin_ids(sequence, tokenizer_output)
            pinyin_ids_arr.append(torch.tensor(pinyin_ids))
        padded_pinyin_ids = pad_sequence(pinyin_ids_arr, batch_first=True, padding_value=0)
        return padded_pinyin_ids
    
    def convert_sentence_to_pinyin_ids(self, sentence, tokenizer_output): # 一個句子而已    
            with open('ChineseBERT-base\config\pinyin_map.json', encoding='utf8') as fin:
                pinyin_dict = json.load(fin)
            # load char id map tensor
            with open('ChineseBERT-base\config\id2pinyin.json', encoding='utf8') as fin:
                id2pinyin = json.load(fin)
            # load pinyin map tensor
            with open('ChineseBERT-base\config\pinyin2tensor.json', encoding='utf8') as fin:
                pinyin2tensor = json.load(fin)
                
            # get pinyin of a sentence
            pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
            pinyin_locs = {}
            # get pinyin of each location
            for index, item in enumerate(pinyin_list):
                pinyin_string = item[0]
                if pinyin_string == "not chinese":
                    continue
                if pinyin_string in pinyin2tensor:
                    pinyin_locs[index] = pinyin2tensor[pinyin_string]
                else:
                    ids = [0] * 8
                    for i, p in enumerate(pinyin_string):
                        if p not in pinyin_dict["char2idx"]:
                            ids = [0] * 8
                            break
                        ids[i] = pinyin_dict["char2idx"][p]
                    pinyin_locs[index] = ids

            pinyin_ids = []
            for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens(), tokenizer_output['offset_mapping'])):
                if offset[1] - offset[0] != 1:
                    pinyin_ids.append([0] * 8)
                    continue
                if offset[0] in pinyin_locs:
                    pinyin_ids.append(pinyin_locs[offset[0]])
                else:
                    pinyin_ids.append([0] * 8)

            return pinyin_ids # list[List[]]
    
    def cal_error_weight(self, pred_ids, original_ids): 
        error_sentences = []
        correct_sentences = []

        with open('traditional_wang127k.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        for d in data:
            error_sentences.append(d['source'])
            correct_sentences.append(d['target'])
                    
        error2correct = {}

        for i in range(len(error_sentences)):
            for e_word, c_word in zip(error_sentences[i], correct_sentences[i]):
                if e_word != c_word:
                    if e_word not in error2correct:
                        error2correct[e_word] = []
                    error2correct[e_word].append(c_word)

        error2correct_num = {}
        for k, correct_word in error2correct.items():
            cal_num = {}
            for word in correct_word:
                if word not in cal_num:
                    cal_num[word] = 1
                else:
                    cal_num[word] += 1
            error2correct_num[k] = cal_num

        def error_probabilities(query):
            if query not in error2correct_num:
                return {}
            total = 0
            for v in error2correct_num[query].values():
                total += v
            return {key: value/total for key, value in error2correct_num[query].items()}
        
        # find typo
        bs, _ = pred_ids.shape
        pred_ids = pred_ids.tolist()
        original_ids = original_ids.tolist()
        mask_arr = []
        for row in range(bs):
            mask = np.ones(23236)
            for p_ids, o_ids in zip(pred_ids[row][1:], original_ids[row][1:]):
                if o_ids != 102:
                    if p_ids != o_ids:
                        o_token = self.tokenizer.convert_ids_to_tokens(o_ids)
                        probability_dict = error_probabilities(o_token)
                        for k, v in probability_dict.items():
                            mask_idx = self.tokenizer.convert_tokens_to_ids(k)
                            mask[mask_idx] += v
            mask_arr.append(mask)
        mask_arr = torch.tensor(mask_arr, dtype=torch.float32)
        
        return mask_arr 

class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
    def forward(self, word_embeddings, summary_embeddings):
        attn_output, _ = self.attention(
            summary_embeddings,
            word_embeddings,
            word_embeddings
        )
        return attn_output
    
class DetectionNetwork(nn.Module):

    def __init__(self):
        super(DetectionNetwork, self).__init__()
        self.bert = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
        self.word_embedding_table = self.bert.get_input_embeddings()
        hidden_size = self.bert.config.hidden_size
        self.cross_attention = CrossAttention(hidden_size)
        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)
        
        self.position_embeddings = self.bert.embeddings.position_embeddings
        self.transformer_blocks = copy.deepcopy(self.bert.encoder.layer[:2])
        
        self.dropout = nn.Dropout(0.1)

        self.dense_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.init_weights()
    
    def init_weights(self):
        init.xavier_uniform_(self.fusion_layer.weight)

    def forward(self, sequences, summaries, max_length):
        inputs, word_embeddings, summary_embeddings = self.get_inputs_and_word_embeddings(sequences, summaries, max_length)
        word_embeddings = self.dropout(word_embeddings)
        summary_embeddings = self.dropout(summary_embeddings)
        
        summary_context = self.cross_attention(word_embeddings, summary_embeddings)
        fused_embeddings = torch.cat([word_embeddings, summary_context], dim=-1)
        input_embeds = self.fusion_layer(fused_embeddings)
        sequence_length = word_embeddings.size(1)
        position_embeddings = self.position_embeddings(torch.LongTensor(range(sequence_length)).to(device))
        x = input_embeds + position_embeddings
        for transformer_layer in self.transformer_blocks:
            x = transformer_layer(x)[0]

        hidden_states = x
        return hidden_states, self.dense_layer(hidden_states).squeeze(2) * inputs['attention_mask'] # 傳去correction, 算loss
    
    def get_inputs_and_word_embeddings(self, sequences, summaries, max_length=128):
        inputs = self.tokenizer(sequences, padding='max_length', max_length=max_length, return_tensors='pt',
                                truncation=True).to(device)
        summary_tokens = self.tokenizer(summaries, padding='max_length', max_length=max_length, return_tensors='pt',
                                truncation=True).to(device)
        word_embeddings = self.word_embedding_table(inputs['input_ids'])
        summary_embeddings = self.word_embedding_table(summary_tokens['input_ids'])
        summary_embeddings = summary_embeddings * summary_tokens['attention_mask'].unsqueeze(-1)
        return inputs, word_embeddings, summary_embeddings

class SummaryBert(nn.Module):

    def __init__(self):
        super(SummaryBert, self).__init__()
        self.correction_network = CorrectionNetwork()
        self._init_correction_dense_layer()
        self.detection_network = DetectionNetwork()

    def forward(self, sequences, summaries, max_length=128):
        hidden_states, detection_outputs = self.detection_network(sequences, summaries, max_length)
        
        correction_outputs = self.correction_network(sequences, hidden_states)
        return correction_outputs, detection_outputs

    def _init_correction_dense_layer(self):
        self.correction_network.dense_layer.weight.data = self.correction_network.word_embedding_table.weight.data
    
model = SummaryBert()
class Loss(nn.Module):
    def __init__(self, coefficient=0.85):
        super(Loss, self).__init__()
        self.correction_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.detection_criterion = nn.BCELoss()
        self.coefficient = coefficient

    def forward(self, correction_outputs, correction_targets, detection_outputs, detection_targets):
        correction_loss = self.correction_criterion(correction_outputs.view(-1, correction_outputs.size(2)),
                                                    correction_targets.view(-1))
        detection_loss = self.detection_criterion(detection_outputs, detection_targets)
        return self.coefficient * correction_loss + (1 - self.coefficient) * detection_loss

class CSCDataset(Dataset):

    def __init__(self):
        super(CSCDataset, self).__init__()
        with open("newVer_wang_summary.json", encoding='utf-8') as f:
            train_data = json.load(f)

        self.train_data = train_data

    def __getitem__(self, index):
        src = self.train_data[index]['source']
        tgt = self.train_data[index]['target']
        summary = self.train_data[index]['summary']
        return src, tgt, summary

    def __len__(self):
        return len(self.train_data)

train_data = CSCDataset()
def collate_fn(batch):
    src, tgt, summary = zip(*batch)
    src, tgt, summary = list(src), list(tgt), list(summary)

    d_tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    d_src_tokens = d_tokenizer(src, padding='max_length', max_length=128, return_tensors='pt', truncation=True)['input_ids']
    d_tgt_tokens = d_tokenizer(tgt, padding='max_length', max_length=128, return_tensors='pt', truncation=True)['input_ids']
    
    c_tokenizer = BertTokenizerFast('ChineseBERT-base/vocab.txt')
    c_src_tokens = c_tokenizer(src, padding='max_length', max_length=128, return_tensors='pt', truncation=True)['input_ids']
    c_tgt_tokens = c_tokenizer(tgt, padding='max_length', max_length=128, return_tensors='pt', truncation=True)['input_ids']

    correction_targets = c_tgt_tokens
    detection_targets = (d_src_tokens != d_tgt_tokens).float()
    return src, correction_targets, detection_targets, c_src_tokens, summary

train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

criterion = Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

start_epoch = 0
total_step = 0

if os.path.exists(model_path):
    if not torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location='cpu')
    else:
        checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    total_step = checkpoint['total_step']
    print("恢復訓練，epoch:", start_epoch)
    
model = model.to(device)
model = model.train()
total_loss = 0.

d_recall_numerator = 0
d_recall_denominator = 0
d_precision_numerator = 0
d_precision_denominator = 0
c_recall_numerator = 0
c_recall_denominator = 0
c_precision_numerator = 0
c_precision_denominator = 0

for epoch in range(start_epoch, epochs):
    step = 0

    for sequences, correction_targets, detection_targets, correction_inputs, summary in train_loader:
        correction_targets, detection_targets = correction_targets.to(device), detection_targets.to(device)
        correction_inputs = correction_inputs.to(device)
        correction_outputs, detection_outputs = model(sequences, summary)
        loss = criterion(correction_outputs, correction_targets, detection_outputs, detection_targets)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += 1
        total_step += 1

        total_loss += loss.detach().item()

        d_predicts = detection_outputs >= 0.5
        d_recall_numerator += d_predicts[detection_targets == 1].sum().item()
        d_recall_denominator += (detection_targets == 1).sum().item()
        d_precision_denominator += d_predicts.sum().item()
        d_precision_numerator += (detection_targets[d_predicts == 1]).sum().item()

        correction_outputs = correction_outputs.argmax(2)
        correction_outputs[(correction_targets == 0) | (correction_targets == 101) | (correction_targets == 102)] = 0
        correction_targets[(correction_targets == 101) | (correction_targets == 102)] = 0
        c_predicts = correction_outputs == correction_targets
        c_recall_numerator += c_predicts[detection_targets == 1].sum().item()
        c_recall_denominator += (detection_targets == 1).sum().item()
        correction_inputs[(correction_inputs == 101) | (correction_inputs == 102)] = 0
        c_precision_denominator += (correction_outputs != correction_inputs).sum().item()
        c_precision_numerator += c_predicts[correction_outputs != correction_inputs].sum().item()

        if total_step % log_after_step == 0:
            loss = total_loss / log_after_step
            d_recall = d_recall_numerator / (d_recall_denominator + 1e-9)
            d_precision = d_precision_numerator / (d_precision_denominator + 1e-9)
            c_recall = c_recall_numerator / (c_recall_denominator + 1e-9)
            c_precision = c_precision_numerator / (c_precision_denominator + 1e-9)

            print("Epoch {}, "
                  "Step {}/{}, "
                  "Total Step {}, "
                  "loss {:.5f}, "
                  "detection recall {:.4f}, "
                  "detection precision {:.4f}, "
                  "correction recall {:.4f}, "
                  "correction precision {:.4f}".format(epoch, step, len(train_loader), total_step,
                                                       loss,
                                                       d_recall,
                                                       d_precision,
                                                       c_recall,
                                                       c_precision))

            total_loss = 0.
            total_correct = 0
            total_num = 0
            d_recall_numerator = 0
            d_recall_denominator = 0
            d_precision_numerator = 0
            d_precision_denominator = 0
            c_recall_numerator = 0
            c_recall_denominator = 0
            c_precision_numerator = 0
            c_precision_denominator = 0

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'total_step': total_step,
    }, model_path)
