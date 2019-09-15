import os
import pickle
import json
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from nltk.stem import WordNetLemmatizer
from torchtext import data
import pandas as pd
#import graph.data_generate as dg

flatten = lambda l: [item for sublist in l for item in sublist]
wnl = WordNetLemmatizer()

def to_bert(tokenizer, text, tags):
    text_bert = []
    tags_bert = []
    for word, tag in zip(text, tags):
        if len(tokenizer.tokenize(word)):
            text_bert.append(word)
            tags_bert.append(tag)
    assert len(text_bert) == len(tags_bert)
    return text_bert, tags_bert



class Example():
    def __init__(self, words, tags, intent):
        self.words = words
        self.tags = tags
        self.intent = intent

class BertExample():
    def __init__(self, words, tags, intent, o_words):
        self.words = words
        self.tags = tags
        self.intent = intent
        self.o_words = o_words

def preprocessing(file_path, output_file, is_training=True):
    """
    atis-2.train.w-intent.iob
    """
    try:
        train = open(file_path, "r").readlines()
        print("Successfully load data. # of set : %d " % len(train))
    except:
        print("No such file!")
        return None, None, None, None

    try:
        train = [t[:-1] for t in train]
        train = [[t.split("\t")[0].split(" "), list(filter(lambda item: item, t.split("\t")[1].split(" ")[:-1])), t.split("\t")[1].split(" ")[-1]] for t in train]
        train = [[t[0][1:-1], t[1][1:], t[2]] for t in train]

        seq_in, seq_out, intent = list(zip(*train))
        vocab = set(flatten(seq_in))
        slot_tag = set(flatten(seq_out))
        intent_tag = set(intent)
        print("# of vocab : {vocab}, # of slot_tag : {slot_tag}, # of intent_tag : {intent_tag}".format(vocab=len(vocab),
                                                                                                      slot_tag=len(
                                                                                                          slot_tag),
                                                                                                      intent_tag=len(
                                                                                                          intent_tag)))
    except:
        print("Please, check data format! It should be 'raw sentence \t BIO tag sequence intent'. The following is a sample.")
        print("BOS i want to fly from baltimore to dallas round trip EOS\tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight")
        return None, None, None, None


    word2index = {'<PAD>': 0, '<UNK>': 1}
    # for token in vocab:
    #     if token not in word2index.keys():
    #         word2index[token] = len(word2index)
    #
    # tag2index = {'<PAD>': 0}
    # for tag in slot_tag:
    #     if tag not in tag2index.keys():
    #         tag2index[tag] = len(tag2index)
    #
    # intent2index = {}
    # for ii in intent_tag:
    #     if ii not in intent2index.keys():
    #         intent2index[ii] = len(intent2index)

    examples = []
    words_len = []
    for words, tags, ini in tqdm(zip(seq_in, seq_out, intent)):
        assert len(words) == len(tags), 'word : {}, tags : {}'.format(words , tags)
        words = list(map(lambda word:dg.re_func(wnl.lemmatize(word.lower())), words))
        e = Example(words, tags, ini)
        examples.append(e)
        words_len.append(len(words))
    print('data len : {}'.format(len(examples)))
    print('max len : {}, avg len : {}, min len : {}'.format(max(words_len), sum(words_len) / len(words_len), min(words_len)))
    pickle.dump(examples, open(os.path.join('data_dir',output_file),'wb'))
    #if is_training:
    #    json.dump(word2index, open(os.path.join('data_dir','word_vocab.txt'),'w', encoding='utf-8'), indent=2,  ensure_ascii=False)
    #    json.dump(tag2index, open(os.path.join('data_dir', 'train_tag_vocab.txt'),'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    #    json.dump(intent2index, open(os.path.join('data_dir', 'train_intent_vocab.txt'),'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    #else:
    #    intent_dict = json.load(open(os.path.join('data_dir', 'intent_vocab.txt'),'r', encoding='utf-8'))

        # for key in intent2index.keys():
        #     if key not in intent_dict:
        #         print(f'intent new key : {key}, len : {len(intent_dict)}')
        #         intent_dict[key] = len(intent_dict)
        #
        # slot_dict = json.load(open(os.path.join('data_dir', 'tag_vocab.txt'),'r', encoding='utf-8'))
        #
        # for key in tag2index.keys():
        #     if key not in slot_dict:
        #         print(f'slot new key : {key}, len : {len(slot_dict)}')
        #         slot_dict[key] = len(slot_dict)
        # json.dump(slot_dict, open(os.path.join('data_dir', 'tag_vocab.txt'), 'w', encoding='utf-8'), indent=2,
        #           ensure_ascii=False)
        # json.dump(intent_dict, open(os.path.join('data_dir', 'intent_vocab.txt'), 'w', encoding='utf-8'), indent=2,
        #           ensure_ascii=False)

def bert_preprocessing(file_path, output_file):
    tokenizer = BertTokenizer('bert/vocab.txt')
    try:
        train = open(file_path, "r").readlines()
        print("Successfully load data. # of set : %d " % len(train))
    except:
        print("No such file!")
        return None, None, None, None

    try:
        train = [t[:-1] for t in train]
        train = [[t.split("\t")[0].split(" "), list(filter(lambda item: item, t.split("\t")[1].split(" ")[:-1])),
                  t.split("\t")[1].split(" ")[-1]] for t in train]
        train = [[t[0][1:-1], t[1][1:], t[2]] for t in train]

        seq_in, seq_out, intent = list(zip(*train))
        vocab = set(flatten(seq_in))
        slot_tag = set(flatten(seq_out))
        intent_tag = set(intent)
        print("# of vocab : {vocab}, # of slot_tag : {slot_tag}, # of intent_tag : {intent_tag}".format(
            vocab=len(vocab),
            slot_tag=len(
                slot_tag),
            intent_tag=len(
                intent_tag)))
    except:
        print(
            "Please, check data format! It should be 'raw sentence \t BIO tag sequence intent'. The following is a sample.")
        print(
            "BOS i want to fly from baltimore to dallas round trip EOS\tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight")
        return None, None, None, None

    examples = []
    words_len = []
    for words, tags, ini in tqdm(zip(seq_in, seq_out, intent)):
        assert len(words) == len(tags), 'word : {}, tags : {}'.format(words, tags)
        words = list(map(lambda a:a.lower(), words))
        #stem
        o_words = words
        words, tags = to_bert(tokenizer, words, tags)
        e = BertExample(words, tags, ini, o_words)
        examples.append(e)
        words_len.append(len(words))
    print('data len : {}'.format(len(examples)))
    print('max len : {}, avg len : {}, min len : {}'.format(max(words_len), sum(words_len) / len(words_len),
                                                            min(words_len)))

    pickle.dump(examples, open('data_dir/'+output_file, 'wb'))


def snip_bert_preprocessing(file_path, output_file):
    '''
    text, tags, intent
    '''
    tokenizer = BertTokenizer('bert/vocab.txt')
    train = pickle.load(open(file_path, 'rb'))
    print("Successfully load data. # of set : %d " % len(train))

    seq_in, seq_out, intent = list(zip(*train))

    examples = []
    words_len = []
    for words, tags, ini in tqdm(zip(seq_in, seq_out, intent)):
        assert len(words) == len(tags), 'word : {}, tags : {}'.format(words, tags)
        words = list(map(lambda a:a.lower(), words))
        #stem
        o_words = words
        words, tags = to_bert(tokenizer, words, tags)
        e = BertExample(words, tags, ini, o_words)
        examples.append(e)
        words_len.append(len(words))
    print('data len : {}'.format(len(examples)))
    print('max len : {}, avg len : {}, min len : {}'.format(max(words_len), sum(words_len) / len(words_len),
                                                            min(words_len)))

    pickle.dump(examples, open('snip/'+output_file, 'wb'))


def build_vocab():
    train_data = pickle.load(open('data_dir/train.pkl','rb'))
    test_data = pickle.load(open('data_dir/test.pkl','rb'))
    word2index = {'<PAD>': 0, '<UNK>': 1}
    for data in train_data:
        words = data.words
        for word in words:
            if word not in word2index:
                word2index[word] = len(word2index)
    for data in test_data:
        words = data.words
        for word in words:
            if word not in word2index:
                word2index[word] = len(word2index)
    print(len(word2index))
    json.dump(word2index, open(os.path.join('data_dir', 'word_vocab.txt'), 'w', encoding='utf-8'), indent=2,
              ensure_ascii=False)


import numpy as np
import torch
# def get_glove_emb(glove_file):
#     d = {}
#     unk = None
#     with open(glove_file) as fw:
#         for line in fw.readlines():
#             row = line.strip().split()
#             word = ' '.join(row[:-300])
#             vector = np.array(list(map(lambda r:float(r), row[-300:])))
#
#             print(word)
#             print(len(vector))
#             if unk is None:
#                 unk = vector
#             else:
#                 unk += vector
#             assert word not in d
#             d[word] = vector
#     return d, unk / len(d)


def get_glove_emb(glove_file):
    df = pd.read_csv(glove_file, sep=' ', quoting=3, header=None, index_col=0)
    #glove = {key:val.values for key,val in df.T.items()}
    #pickle.dump(glove, open('glove.pkl','wb'))
    glove = pickle.load(open('glove.pkl','rb'))
    unk_emb = np.mean(np.array(list(glove.values())), axis=0)
    return glove, unk_emb

def get_pretrain_embedding(glove_file):
    glove_emb, unk_emb = get_glove_emb(glove_file)
    print('glove finished.......................................')
    vocab = json.load(open(os.path.join('data_dir', 'word_vocab.txt')))
    pre_emb = np.zeros(shape=[len(vocab), 300], dtype=float)
    for i, key in enumerate(vocab.keys()):
        if key == '<PAD>':
            continue
        if key == '<UNK>' or key not in glove_emb:
            print(f'key : {key} not exist')
            pre_emb[i] =unk_emb
        else:
            pre_emb[i] = glove_emb[key]
    pre_emb = torch.FloatTensor(pre_emb)
    pickle.dump(pre_emb, open('data_dir/pre_emb.pkl','wb'))

if __name__ == '__main__':
    #get_pretrain_embedding('glove.840B.300d.txt')
    #get_glove_emb('glove.840B.300d.txt')
    build_vocab()
    #preprocessing(os.path.join('atis','atis.train.w-intent.iob'), 'train.pkl', is_training=True)
    #bert_preprocessing(os.path.join('atis','atis.test.w-intent.iob'),'test_manual_bert.pkl')
    #snip_bert_preprocessing('snip/snip_train.pkl', 'snip_train_bert.pkl')