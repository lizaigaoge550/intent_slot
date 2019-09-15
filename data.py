from torchvision import transforms
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from data_preprocess import Example
from intent_manual_feature import MaualBertExample
import pickle
from utils import tokens_to_indices
import json

class Vocab_Json():
    def __init__(self, vocab_file):
        self.word_to_idx = json.load(open(vocab_file, 'r'))
        self.id_to_word = self.id_word_map()

    def id_word_map(self):
        d = {}
        for key,value in self.word_to_idx.items():
            d[value] = key
        return d

    def word2id(self, word, type='token'):
        if type == 'token':
            return self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
        else:
            return self.word_to_idx[word]
    def id2word(self, id):
        # if id not in self.id_to_word:
        #     return len(self.id_to_word) + 1
        return self.id_to_word[id]

    def __len__(self):
        return len(self.word_to_idx)


class Vocab():
    def __init__(self, vocab_file):
        self.word_to_idx = {}
        self.id_to_word = {}
        with open(vocab_file, 'r') as fr:
            for line in fr.readlines():
                word = line.split('\n')[0]
                if word in self.word_to_idx:
                    raise ("Duplicate word : {} exist".format(word))
                self.word_to_idx[word] = len(self.word_to_idx)
                self.id_to_word[len(self.word_to_idx)-1] = word

    def word2id(self, word, type='token'):
        if type == 'token':
            return self.word_to_idx.get(word, self.word_to_idx['[UNK]'])
        else:
            return self.word_to_idx[word]
    def id2word(self, id):
        return self.id_to_word[id]

    def __len__(self):
        return len(self.word_to_idx)


class DataSet(object):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = self.transform(sample)
        return sample


class Tensor(object):
    def __init__(self, token_vocab, label_vocab, intent_vocab):
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.intent_vocab = intent_vocab
    def __call__(self, sample):
        tokens = sample.words
        tags = sample.tags
        intent = sample.intent
        #word to id
        token_ids = [self.token_vocab.word2id(word) for word in tokens]
        tag_ids = [self.label_vocab.word2id(tag, type='label') for tag in tags]
        intent_id = self.intent_vocab.word2id(intent, type='label')
        assert len(token_ids) == len(tag_ids)
        return {'tokens':token_ids, 'tags':tag_ids, 'intent':intent_id}


class XlNetTensor(object):
    def __init__(self, token_vocab, label_vocab, intent_vocab):
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.intent_vocab = intent_vocab
    def __call__(self, sample):
        tokens_embed = sample['vector']
        tags = sample['tags']
        intent = sample['intent']
        #word to id
        tag_ids = [self.label_vocab.word2id(tag, type='label') for tag in tags]
        intent_id = self.intent_vocab.word2id(intent, type='label')
        assert len(tokens_embed) == len(tag_ids)
        return {'tokens_embed':tokens_embed, 'tags':tag_ids, 'intent':intent_id}


class BertTensor(object):
    def __init__(self, token_vocab, label_vocab, intent_vocab, is_training=True, **kwargs):
        self.kwargs = kwargs
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.intent_vocab = intent_vocab
        self.tokenizer = BertTokenizer(kwargs['vocab_path'])
        self.is_training = is_training

    def chunk(self, word_idx, offset, max_seq_len):
        while len(offset) > max_seq_len:
            idx = offset.pop()
        return word_idx[:idx], offset

    def __call__(self, sample):
        max_len = self.kwargs.get('max_token_lens', 512)
        word_idx, offset = tokens_to_indices(sample.words, vocab=self.token_vocab, tokenizer=self.tokenizer, max_pieces=max_len)
        assert len(offset) == len(set(offset)), '{}, {}'.format(len(offset), len(set(offset)))
        if self.is_training:
            tags = [self.label_vocab.word2id(tag, type='label') for tag in sample.tags]
            indent = self.intent_vocab.word2id(sample.intent, type='label')
        #    if len(tags) > len(offset):
        #        tags = tags[:len(offset)]
            assert len(offset) == len(tags), f'{offset}, {tags}'
            return {'tokens': word_idx, 'tags':tags, 'offset': offset,  'origin_tokens':sample.words[:len(offset)], 'intent':indent}
        else:
            return {'tokens': word_idx, 'offset': offset, 'origin_tokens': sample.words[:len(offset)]}

class ManualBertTensor(object):
    def __init__(self, token_vocab, label_vocab, intent_vocab, is_training=True, **kwargs):
        self.kwargs = kwargs
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.intent_vocab = intent_vocab
        self.tokenizer = BertTokenizer(kwargs['vocab_path'])
        self.is_training = is_training

    def chunk(self, word_idx, offset, max_seq_len):
        while len(offset) > max_seq_len:
            idx = offset.pop()
        return word_idx[:idx], offset

    def __call__(self, sample):
        max_len = self.kwargs.get('max_token_lens', 512)
        word_idx, offset = tokens_to_indices(sample.words, vocab=self.token_vocab, tokenizer=self.tokenizer, max_pieces=max_len)
        assert len(offset) == len(set(offset)), '{}, {}'.format(len(offset), len(set(offset)))
        tags = [self.label_vocab.word2id(tag, type='label') for tag in sample.tags]
        indent = self.intent_vocab.word2id(sample.intent, type='label')
        assert len(offset) == len(tags), f'{offset}, {tags}'
        return {'tokens': word_idx, 'tags':tags, 'offset': offset,  'origin_tokens':sample.words[:len(offset)],
                'intent':indent, 'manual_feature':sample.manual_features}


def loading_data(data_path, token_vocab, intent_vocab, tag_vocab, vocab_path=None, net_type='normal'):
    datas = pickle.load(open(data_path,'rb'))
    print('loading dataset finished......................')
    if net_type == 'bert':
        return DataSet(datas, transform=transforms.Compose([BertTensor(token_vocab, label_vocab=tag_vocab,
                                                                       intent_vocab=intent_vocab, vocab_path=vocab_path)]))

    elif net_type == 'xlnet':
        return DataSet(datas, transform=transforms.Compose([XlNetTensor(token_vocab, tag_vocab, intent_vocab)]))
    else:
        return DataSet(datas, transform=transforms.Compose([Tensor(token_vocab, tag_vocab, intent_vocab)]))



