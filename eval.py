import os
import copy
import json
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

from torch import nn
from transformers import BertTokenizerFast
from model.chineseBert import Bert
from pypinyin import pinyin, Style
from transformers import AutoModel, AutoTokenizer
import torch.nn.init as init
from tqdm import tqdm

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
        input_ids = self.tokenizer(sequences, padding='max_length', max_length=128, return_tensors='pt',
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

        with open('sighan_all.json', 'r', encoding='utf-8') as f:
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

class MDCSpellModel(nn.Module):

    def __init__(self):
        super(MDCSpellModel, self).__init__()
        self.correction_network = CorrectionNetwork()
        self._init_correction_dense_layer()
        self.detection_network = DetectionNetwork()

    def forward(self, sequences, summaries, max_length=128):
        hidden_states, detection_outputs = self.detection_network(sequences, summaries, max_length)
        
        correction_outputs = self.correction_network(sequences, hidden_states)
        return correction_outputs, detection_outputs

    def _init_correction_dense_layer(self):
        self.correction_network.dense_layer.weight.data = self.correction_network.word_embedding_table.weight.data
    
model = MDCSpellModel().to(device)
pretrained_model_path = './outputs/summaryBert/finetuned/v1.pt'
checkpoint = torch.load(pretrained_model_path, map_location=device)
model.load_state_dict(checkpoint['model'])
print("加載預訓練模型權重！")

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

def score_f_sent(inputs, golds, preds, ds_version):
    assert len(inputs) == len(golds)
    assert len(golds) == len(preds)
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    fout = open(f'sent_pred_result_{ds_version}.txt', 'w', encoding='utf-8')
    for ori_tags, god_tags, prd_tags in zip(inputs, golds, preds):
        if None in ori_tags or None in god_tags or None in prd_tags:
            continue
        assert len(ori_tags) == len(god_tags)
        assert len(god_tags) == len(prd_tags)
        gold_errs = [idx for (idx, tk) in enumerate(god_tags) if tk != ori_tags[idx]]
        pred_errs = [idx for (idx, tk) in enumerate(prd_tags) if tk != ori_tags[idx]]
        if len(gold_errs) > 0 or len(pred_errs) > 0:
            fout.writelines('\n%s\n%s\n%s\n' % ('|'.join(ori_tags), '|'.join(god_tags),'|'.join(prd_tags)))
        if len(gold_errs) > 0:
            total_gold_err += 1
            fout.writelines('gold_err\n')
        if len(pred_errs) > 0:
            fout.writelines('check_err\n')
            total_pred_err += 1
            if gold_errs == pred_errs:
                check_right_pred_err += 1
                fout.writelines('check_right\n')
            if god_tags == prd_tags:
                right_pred_err += 1
                fout.writelines('correct_right\n')
    fout.close()
    p = 1. * check_right_pred_err / total_pred_err
    r = 1. * check_right_pred_err / total_gold_err
    f = 2 * p * r / (p + r + 1e-13)
    #print(total_gold_err, total_pred_err, right_pred_err, check_right_pred_err)
    print('sent check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    p = 1. * right_pred_err / total_pred_err
    r = 1. * right_pred_err / total_gold_err
    f = 2 * p * r / (p + r + 1e-13)
    print('sent correction: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    return p, r, f

def score_f(ans, ds_version, print_flg=False, only_check=False):
    fout = open(f'pred_{ds_version}.txt', 'w', encoding="utf-8")
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    inputs, golds, preds = ans
    assert len(inputs) == len(golds)
    assert len(golds) == len(preds)
    for ori, god, prd in zip(inputs, golds, preds):
        ori_txt = str(ori)
        god_txt = str(god) #''.join(list(map(str, god)))
        prd_txt = str(prd) #''.join(list(map(str, prd)))
        if print_flg is True:
            print(ori_txt, '\t', god_txt, '\t', prd_txt)
        if 'UNK' in ori_txt:
            continue
        if ori_txt == god_txt and ori_txt == prd_txt:
            continue
        if prd_txt != god_txt:
            fout.writelines('%s\t%s\t%s\n' % (ori_txt, god_txt, prd_txt)) 
        if ori != god:
            total_gold_err += 1
        if prd != ori:
            total_pred_err += 1
        if (ori != god) and (prd != ori):
            check_right_pred_err += 1
            if god == prd:
                right_pred_err += 1
    fout.close()

    #check p, r, f
    p = 1. * check_right_pred_err / (total_pred_err + 0.001)
    r = 1. * check_right_pred_err / (total_gold_err + 0.001)
    f = 2 * p * r / (p + r +  1e-13)
    print('token check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    if only_check is True:
        return p, r, f

    #correction p, r, f
    #p = 1. * right_pred_err / (total_pred_err + 0.001)
    pc = 1. * right_pred_err / (check_right_pred_err + 0.001)
    rc = 1. * right_pred_err / (total_gold_err + 0.001)
    fc = 2 * pc * rc / (pc + rc + 1e-13) 
    print('token correction: p=%.3f, r=%.3f, f=%.3f' % (pc, rc, fc))
    return p, r, f

def sentence_eval(test_data, ds_version):
    inputs = []
    golds = []
    preds = []
    all_inputs = []
    all_golds = []
    all_preds = []

    prograss = tqdm(range(len(test_data)))
    for i in prograss:
        src, tgt, summary = test_data[i]['source'], test_data[i]['target'], test_data[i]['summary']

        src_enc = tokenizer(src, return_tensors='pt', max_length=128, truncation=True)
        tgt_enc = tokenizer(tgt, return_tensors='pt', max_length=128, truncation=True)

        src_tokens = src_enc['input_ids'][0][1:-1]  # 去掉 [CLS] 和 [SEP]
        tgt_tokens = tgt_enc['input_ids'][0][1:-1]

        if len(src_tokens) != len(tgt_tokens):
            print("第%d條資料異常" % i)
            continue

        # 模型預測
        correction_outputs, _ = model([src], summary)
        predict_ids = correction_outputs[0][1:len(src_tokens) + 1].argmax(1).detach().cpu()

        # Tensor to tokens
        ori_tokens = tokenizer.convert_ids_to_tokens(src_tokens)
        gold_tokens = tokenizer.convert_ids_to_tokens(tgt_tokens)
        pred_tokens = tokenizer.convert_ids_to_tokens(predict_ids)

        for l in range(len(ori_tokens)):
            inputs.append(ori_tokens[l])
            golds.append(gold_tokens[l])
            preds.append(pred_tokens[l])

        all_inputs.append(ori_tokens)
        all_golds.append(gold_tokens)
        all_preds.append(pred_tokens)

    # 評分
    score_f_sent(all_inputs, all_golds, all_preds, ds_version)

    score_f((inputs, golds, preds), ds_version, only_check=False)
    
arr = []
ds_version = 13 # 切換不同資料集
with open(f"sighan{ds_version}_summary.json", encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        if item['source'] != item['target']:
            arr.append(item)
sentence_eval(arr, ds_version)