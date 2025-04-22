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
from chineseBert import Bert
from pypinyin import pinyin, Style
from transformers import AutoModel, AutoTokenizer
import torch.nn.init as init
from tqdm.auto import tqdm
import logging

# 句子的长度，作者并没有说明。我这里就按经验取一个
max_length = 128
# 作者使用的batch_size
batch_size = 32
# epoch数，作者并没有具体说明，按经验取一个
epochs = 20

# 每${log_after_step}步，打印一次日志
log_after_step = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
    
class CorrectionNetwork(nn.Module):

    def __init__(self):
        super(CorrectionNetwork, self).__init__()
        # BERT分词器，作者并没提到自己使用的是哪个中文版的bert，我这里就使用一个比较常用的
        self.tokenizer = BertTokenizerFast('ChineseBERT-base/vocab.txt')
        # BERT
        self.bert = Bert.from_pretrained('ChineseBERT-base')
        # BERT的word embedding，本质就是个nn.Embedding
        self.word_embedding_table = self.bert.get_input_embeddings()
        # 预测层。hidden_size是词向量的大小，len(self.tokenizer)是词典大小
        self.dense_layer = nn.Linear(self.bert.config.hidden_size, len(self.tokenizer))

    def forward(self, sequences, detect_hidden_states):
        """
        Correction Network的前向传递
        :param inputs: inputs为tokenizer对中文文本的分词结果，
                       里面包含了token对一个的index，attention_mask等
        :param word_embeddings: 使用BERT的word_embedding对token进行embedding后的结果
        :param detect_hidden_states: Detection Network输出hidden state
        :return: Correction Network对个token的预测结果。
        """
        # 1. 使用bert进行前向传递
        pinyin_ids = self.get_pinyin_ids(sequences).to(device)
        input_ids = self.tokenizer(sequences, padding='max_length', max_length=max_length, return_tensors='pt',
                                truncation=True)['input_ids'].to(device)
        bert_outputs = self.bert(input_ids, pinyin_ids)
        # 2. 将bert的hidden_state和Detection Network的hidden state进行融合。
        hidden_states = bert_outputs['last_hidden_state'] + detect_hidden_states
        # 3. 最终使用全连接层进行token预测
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
        """
        :param position_embeddings: bert的position_embeddings，本质是一个nn.Embedding
        :param transformer: BERT的前两层transformer_block，其是一个ModuleList对象
        """
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

        # 定义最后的预测层，预测哪个token是错误的
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
        # 获取token序列的长度，这里为128
        sequence_length = word_embeddings.size(1)
        # 生成position embedding
        position_embeddings = self.position_embeddings(torch.LongTensor(range(sequence_length)).to(device))
        # 融合work_embedding和position_embedding
        x = input_embeds + position_embeddings
        # 将x一层一层的使用transformer encoder进行向后传递
        for transformer_layer in self.transformer_blocks:
            x = transformer_layer(x)[0]

        # 最终返回Detection Network输出的hidden states和预测结果
        hidden_states = x
        return hidden_states, self.dense_layer(hidden_states).squeeze(2) * inputs['attention_mask'] # 傳去correction, 算loss
    
    def get_inputs_and_word_embeddings(self, sequences, summaries, max_length=128):
        """
        对中文序列进行分词和word embeddings处理
        :param sequences: 中文文本序列。例如: ["鸡你太美", "哎呦，你干嘛！"]
        :param max_length: 文本的最大长度，不足则进行填充，超出进行裁剪。
        :return: tokenizer的输出和word embeddings.
        """
        inputs = self.tokenizer(sequences, padding='max_length', max_length=max_length, return_tensors='pt',
                                truncation=True).to(device)
        summary_tokens = self.tokenizer(summaries, padding='max_length', max_length=max_length, return_tensors='pt',
                                truncation=True).to(device)
        # 使用BERT的work embeddings对token进行embedding，这里得到的embedding并不包含position embedding和segment embedding
        word_embeddings = self.word_embedding_table(inputs['input_ids'])
        summary_embeddings = self.word_embedding_table(summary_tokens['input_ids'])
        summary_embeddings = summary_embeddings * summary_tokens['attention_mask'].unsqueeze(-1)
        return inputs, word_embeddings, summary_embeddings

class MDCSpellModel(nn.Module):

    def __init__(self):
        super(MDCSpellModel, self).__init__()
        # 构造Correction Network
        self.correction_network = CorrectionNetwork()
        self._init_correction_dense_layer()
        # 构造Detection Network
        self.detection_network = DetectionNetwork()

    def forward(self, sequences, summaries, max_length=128):
        # Detection Network进行前向传递，获取输出的Hidden State和预测结果
        hidden_states, detection_outputs = self.detection_network(sequences, summaries, max_length)
        
        # Correction Network进行前向传递，获取其预测结果
        correction_outputs = self.correction_network(sequences, hidden_states)
        # 返回Correction Network 和 Detection Network 的预测结果。
        # 在计算损失时`[PAD]`token不需要参与计算，所以这里将`[PAD]`部分全都变为0
        return correction_outputs, detection_outputs

    def _init_correction_dense_layer(self):
        """
        原论文中提到，使用Word Embedding的weight来对Correction Network进行初始化
        """
        self.correction_network.dense_layer.weight.data = self.correction_network.word_embedding_table.weight.data
    
model = MDCSpellModel().to(device)
        
class MDCSpellLoss(nn.Module):
    def __init__(self, coefficient=0.85):
        super(MDCSpellLoss, self).__init__()
        # 定义Correction Network的Loss函数
        self.correction_criterion = nn.CrossEntropyLoss(ignore_index=0)
        # 定义Detection Network的Loss函数，因为是二分类，所以用Binary Cross Entropy
        self.detection_criterion = nn.BCELoss()
        # 权重系数
        self.coefficient = coefficient

    def forward(self, correction_outputs, correction_targets, detection_outputs, detection_targets):
        """
        :param correction_outputs: Correction Network的输出，Shape为(batch_size, sequence_length, hidden_size)
        :param correction_targets: Correction Network的标签，Shape为(batch_size, sequence_length)
        :param detection_outputs: Detection Network的输出，Shape为(batch_size, sequence_length)
        :param detection_targets: Detection Network的标签，Shape为(batch_size, sequence_length)
        :return:
        """
        # 计算Correction Network的loss，因为Shape维度为3，所以要把batch_size和sequence_length进行合并才能计算
        correction_loss = self.correction_criterion(correction_outputs.view(-1, correction_outputs.size(2)),
                                                    correction_targets.view(-1))
        # 计算Detection Network的loss
        detection_loss = self.detection_criterion(detection_outputs, detection_targets)
        # 对两个loss进行加权平均
        return self.coefficient * correction_loss + (1 - self.coefficient) * detection_loss

class CSCDataset(Dataset):

    def __init__(self):
        super(CSCDataset, self).__init__()
        with open("newVer_sighan_summary.json", encoding='utf-8') as f:
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
    return src, correction_targets, detection_targets, c_src_tokens, summary  # src_tokens在计算Correction的精准率时要用到
train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
criterion = MDCSpellLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
start_epoch = 0  # 从哪个epoch开始
total_step = 0  # 一共更新了多少次参数

pretrained_model_path = './drive/MyDrive/models/summaryBertWChineseBert/summaryChineseBertWDetect-model.pt'
checkpoint = torch.load(pretrained_model_path, map_location=device)
model.load_state_dict(checkpoint['model'])
print("成功加载预训练模型权重！")

model = model.to(device)
model = model.train()
total_loss = 0.  # 记录loss

d_recall_numerator = 0  # Detection的Recall的分子
d_recall_denominator = 0  # Detection的Recall的分母
d_precision_numerator = 0  # Detection的precision的分子
d_precision_denominator = 0  # Detection的precision的分母
c_recall_numerator = 0  # Correction的Recall的分子
c_recall_denominator = 0  # Correction的Recall的分母
c_precision_numerator = 0  # Correction的precision的分子
c_precision_denominator = 0  # Correction的precision的分母

def setup_logger(log_dir, log_file='train.log'):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)

    # 避免重複加入 handler
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

logger = setup_logger('./logs')

def score_f_sent(inputs, golds, preds):
    assert len(inputs) == len(golds)
    assert len(golds) == len(preds)
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    fout = open('sent_pred_result.txt', 'w', encoding='utf-8')
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
    logger.info('sent check: p=%.3f, r=%.3f, f=%.3f', p, r, f)

    p = 1. * right_pred_err / total_pred_err
    r = 1. * right_pred_err / total_gold_err
    f = 2 * p * r / (p + r + 1e-13)
    print('sent correction: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    logger.info('sent correction: p=%.3f, r=%.3f, f=%.3f', p, r, f)

    return p, r, f


def score_f(ans, print_flg=False, only_check=False):
    fout = open('pred.txt', 'w', encoding="utf-8")
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
    logger.info('token check: p=%.3f, r=%.3f, f=%.3f', p, r, f)
    if only_check is True:
        return p, r, f

    #correction p, r, f
    #p = 1. * right_pred_err / (total_pred_err + 0.001)
    pc = 1. * right_pred_err / (check_right_pred_err + 0.001)
    rc = 1. * right_pred_err / (total_gold_err + 0.001)
    fc = 2 * pc * rc / (pc + rc + 1e-13) 
    print('token correction: p=%.3f, r=%.3f, f=%.3f' % (pc, rc, fc))
    logger.info('token correction: p=%.3f, r=%.3f, f=%.3f', pc, rc, fc)

    return p, r, f

def sentence_eval(test_data):
    inputs = []
    golds = []
    preds = []
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    prograss = tqdm(range(len(test_data)))
    for i in prograss:
        src, tgt, summary = test_data[i]['source'], test_data[i]['target'], test_data[i]['summary']

        src_enc = tokenizer(src, return_tensors='pt', max_length=128, truncation=True)
        tgt_enc = tokenizer(tgt, return_tensors='pt', max_length=128, truncation=True)

        src_tokens = src_enc['input_ids'][0][1:-1]  # 去掉 [CLS] 和 [SEP]
        tgt_tokens = tgt_enc['input_ids'][0][1:-1]

        if len(src_tokens) != len(tgt_tokens):
            print("第%d条数据异常" % i)
            continue

        # 模型預測
        correction_outputs, _ = model([src], summary)
        predict_ids = correction_outputs[0][1:len(src_tokens) + 1].argmax(1).detach().cpu()

        # Tensor to tokens
        ori_tokens = tokenizer.convert_ids_to_tokens(src_tokens)
        gold_tokens = tokenizer.convert_ids_to_tokens(tgt_tokens)
        pred_tokens = tokenizer.convert_ids_to_tokens(predict_ids)

        inputs.append(ori_tokens)
        golds.append(gold_tokens)
        preds.append(pred_tokens)

    # 評分
    score_f_sent(inputs, golds, preds)

    score_f((inputs, golds, preds), only_check=False)

for i in range(3):

    # 模型存放的位置。
    model_path = './drive/MyDrive/models/summaryBertWChineseBert-finetuned/'
    os.makedirs(model_path, exist_ok=True)
    model_path = model_path + f'summaryChineseBertWDetect-fintuned-model-v{i+1}.pt'
    
    # 恢复之前的训练
    if os.path.exists(model_path):
        if not torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        total_step = checkpoint['total_step']
        print("恢复训练，epoch:", start_epoch)
    print(f'Start training: {i+1}')

    for epoch in range(start_epoch, epochs):
        torch.cuda.empty_cache()
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

            # 计算Detection的recall和precision指标
            # 大于0.5，认为是错误token，反之为正确token
            d_predicts = detection_outputs >= 0.5
            # 计算错误token中被网络正确预测到的数量
            d_recall_numerator += d_predicts[detection_targets == 1].sum().item()
            # 计算错误token的数量
            d_recall_denominator += (detection_targets == 1).sum().item()
            # 计算网络预测的错误token的数量
            d_precision_denominator += d_predicts.sum().item()
            # 计算网络预测的错误token中，有多少是真错误的token
            d_precision_numerator += (detection_targets[d_predicts == 1]).sum().item()

            # 计算Correction的recall和precision
            # 将输出映射成index，即将correction_outputs的Shape由(32, 128, 21128)变为(32,128)
            correction_outputs = correction_outputs.argmax(2)
            # 对于填充、[CLS]和[SEP]这三个token不校验
            correction_outputs[(correction_targets == 0) | (correction_targets == 101) | (correction_targets == 102)] = 0
            # correction_targets的[CLS]和[SEP]也要变为0
            correction_targets[(correction_targets == 101) | (correction_targets == 102)] = 0
            # Correction的预测结果，其中True表示预测正确，False表示预测错误或无需预测
            c_predicts = correction_outputs == correction_targets
            # 计算错误token中被网络正确纠正的token数量
            c_recall_numerator += c_predicts[detection_targets == 1].sum().item()
            # 计算错误token的数量
            c_recall_denominator += (detection_targets == 1).sum().item()
            # 计算网络纠正token的数量
            correction_inputs[(correction_inputs == 101) | (correction_inputs == 102)] = 0
            c_precision_denominator += (correction_outputs != correction_inputs).sum().item()
            # 计算在网络纠正的这些token中，有多少是真正被纠正对的
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

    finetuned_model_path = f'./drive/MyDrive/models/summaryBertWChineseBert-finetuned/summaryChineseBertWDetect-fintuned-model-v{i+1}.pt'
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print("成功加载微調模型权重！")

    arr = []
    with open("sighan13_summary.json", encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if item['source'] != item['target']:
                arr.append(item)
    
    logger.info(f'===== Sighan13 Val-{i+1} =====')
    sentence_eval(arr)

    arr = []
    with open("sighan14_summary.json", encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if item['source'] != item['target']:
                arr.append(item)
    
    logger.info(f'===== Sighan14 Val-{i+1} =====')
    sentence_eval(arr)

    arr = []
    with open("sighan15_summary.json", encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if item['source'] != item['target']:
                arr.append(item)
    
    logger.info(f'===== Sighan15 Val-{i+1} =====')
    sentence_eval(arr)

