#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

from datasets import tqdm
from nltk.corpus import stopwords
import time
import json
from collections import Counter
import numpy as np
import torch
import torch.utils.data
from tools.logger import *
import dgl
from dgl.data.utils import load_graphs
import os
from config import Config
from transformers import BertTokenizer
from nltk import word_tokenize
config = Config()

MAP = {'train': 'train', 'valid': 'val', 'test': 'test'}

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']

FILTERWORD.extend(punctuations)

tokenizer = BertTokenizer.from_pretrained('/root/bert-base-multilingual-cased')

######################################### Example #########################################

class Example(object):
    """Class representing a train/val/test example for single-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents,article_lanauge,sent_max_len, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(strings) for single document or list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        # Store the original strings
        self.original_article_sents = article_sents
        self.original_article_lanauge = article_lanauge
        self.original_abstract = "\n".join(abstract_sents)
        self.tokenizer_lanauge = {
            "en":"en_XX",
            "es":"es_XX",
            "fr":"fr_XX",
            "de":"de_DE"
        }

        # Process the article
        if isinstance(article_sents, list) and isinstance(article_sents[0], list):  # multi document
            self.original_article_sents = []
            for doc in article_sents:
                self.original_article_sents.extend(doc)
        for index,sent in enumerate(self.original_article_sents):
            tokenizer.src_lang = self.tokenizer_lanauge[self.original_article_lanauge[index][str(index)]]
            tokenizer.tgt_lang = "en_XX"
            tokenizers =  tokenizer(sent,max_length=sent_max_len, padding='max_length', truncation=True , return_tensors='pt')
            article_words = tokenizers['input_ids'][0].tolist()
            self.enc_sent_len.append(len(sent.split()))
            self.enc_sent_input_pad.append(article_words)
        self.label = label
        label_shape = (len(self.original_article_sents), len(label))  # [N, len(label)]
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label != []:
            self.label_matrix[np.array(label), np.arange(
                len(label))] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step

    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return:
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)


######################################### ExampleSet #########################################

class ExampleSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self,mode, data_path,cached_features_file, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, bert_path,generator_tokenizer):
        """ Initializes the ExampleSet with the path of data

        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py)
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        """
        self.generator_tokenizer = generator_tokenizer
        self.cached_features_file =cached_features_file

        self.mode = mode
        #self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = readJson(data_path)

        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        self.size = len(self.example_list)

        logger.info("[INFO] Loading filter word File %s", filter_word_path)
        tfidf_w = readText(filter_word_path)
        self.filterwords = FILTERWORD
        #self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
        #self.filterids.append(vocab.word2id("[PAD]"))  # keep "[UNK]" but remove "[PAD]"
        self.filterids =  tokenizer(" ".join(FILTERWORD[0:512]), return_tensors='pt') ['input_ids'][0].tolist()[1:-1]
        self.filterids += tokenizer(" ".join(FILTERWORD[512:0]), return_tensors='pt') ['input_ids'][0].tolist()[1:-1]
        self.filterids.append(0)
        self.tokenizer_lanauge = {
            "en":"en_XX",
            "es":"es_XX",
            "fr":"fr_XX",
            "de":"de_DE"
        }
        lowtfidf_num = 0
        pattern = r"^[0-9]+$"
        word2id_UNK = tokenizer('[UNK]', return_tensors='pt') ['input_ids'][0].tolist()[1:-1]
        for w in tfidf_w:
            word2id = tokenizer(w, return_tensors='pt') ['input_ids'][0].tolist()[1:-1]
            if word2id != word2id_UNK:
                self.filterwords.append(w)
                self.filterids.append(word2id)
                # if re.search(pattern, w) == None:  # if w is a number, it will not increase the lowtfidf_num
                # lowtfidf_num += 1
                lowtfidf_num += 1
            if lowtfidf_num > 5000:
                break

        logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
        self.w2s_tfidf = readJson(w2s_path)

        self.w2s_path = w2s_path
        self.data_path = data_path
        self.bert_path = bert_path
        self.load_features_from_cache()

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"],e["lanauge"],self.sent_max_len, e["label"])
        return example

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        N, m = label_m.shape
        if m < self.doc_max_timesteps:
            pad_m = np.zeros((N, self.doc_max_timesteps - m))
            return np.hstack([label_m, pad_m])
        return label_m

    def AddWordNode(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in self.filterids and wid not in wid2nid.keys():
                    # if wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)

        G.add_nodes(w_nodes)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(w_nodes)
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        G.ndata["dtype"] = torch.zeros(w_nodes)

        return wid2nid, nid2wid

    def CreateGraph(self, input_pad, label, w2s_w, feature_bert=None):
        """ Create a graph for each document

        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, dtype=0
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]

        G.set_e_initializer(dgl.init.zero_initializer)
        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            # id2word_list = tokenizer.decode(c.keys(), skip_special_tokens=True).split(" ")
            for wid in c.keys():
                id2word = tokenizer.decode([wid], skip_special_tokens=True)
                if wid in wid2nid.keys() and id2word in sent_tfw.keys():
                    tfidf = sent_tfw[id2word]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
            G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        G.nodes[sentid2nid].data["bert"] = torch.FloatTensor(feature_bert)

        return G

    def tokenize_generator(self, text, query, summary,lanauge):
        context_input_ids = []
        labels = None
        padded_text = [''] * config.window_size + text + [''] * config.window_size
        context_attention_mask = []
        for turn_id in range(len(text)):
            # turns with a window size.
            #self.generator_tokenizer.src_lang = self.tokenizer_lanauge[lanauge[turn_id][str(turn_id)]]
            self.generator_tokenizer.src_lang ="en_XX"
            self.generator_tokenizer.tgt_lang = "en_XX"
            contextualized_turn = self.generator_tokenizer.eos_token.join(padded_text[turn_id: turn_id + 1 + 2 * config.window_size])

            input_dict = self.generator_tokenizer.prepare_seq2seq_batch(src_texts=contextualized_turn + " // " + query,
                                                                        tgt_texts=summary,
                                                                        max_length=config.max_source_len,
                                                                        max_target_length=config.max_target_len,
                                                                        padding="max_length",
                                                                        truncation=True,
                                                                        )
            context_attention_mask.append(input_dict.attention_mask)
            context_input_ids.append(input_dict.input_ids)
            if labels is None:
                labels = input_dict.labels
            else:
                assert labels == input_dict.labels, '{} != {}'.format(labels, input_dict.labels)

        generator_inputs = {'context_input_ids': context_input_ids,
                            'context_attention_mask': context_attention_mask,
                            'labels': labels}

        if labels is None:
            raise ValueError(text)

        return generator_inputs
    def preprocess(self, inputs):
        session = inputs
        paper = session['text']

        summary = session['summary']

        lanauge = session['lanauge']

        query = ""

        if self.mode == "train" or self.mode == "test":

            oracles =[oracle_id for oracle_id in session['label'] if  len(paper[oracle_id].split(" ")) > 3 ]

        else:
            oracles = []

        generator_inputs = self.tokenize_generator(text=paper, query=query, summary=summary,lanauge=lanauge)

        return oracles, generator_inputs

    def tokenize(self,sent):
        tokens = ' '.join(word_tokenize(sent.lower()))
        return tokens
    def StructGeneratorFeature(self):

        f_dataset = open(self.data_path,'r',encoding='utf-8')
        datasets = f_dataset.readlines()
        features = []
        for session in tqdm(datasets):
            session = json.loads(session)
            features.append(self.preprocess(inputs=(session)))

        return features

    def get_references(self):

        references = []
        for session in tqdm(self.example_list):
            summary = " ".join(session["summary"])
            references.append(self.tokenize(summary))

        return references
    def load_features_from_cache(self):

        if not config.early_preprocess:
            self.cached_features_file = self.cached_features_file + '_late_preprocess'
        print("cached feature file address", self.cached_features_file)
        if os.path.exists(self.cached_features_file) and not config.overwrite_cache:
            print("Loading features from cached file {}".format(self.cached_features_file))
            self.features = torch.load(self.cached_features_file)
        else:
            self.features = self.StructGeneratorFeature()
            print("Saving features into cached file {}".format(self.cached_features_file))
            torch.save(self.features, self.cached_features_file)
    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        label = self.pad_label_m(item.label_matrix)
        w2s_w = self.w2s_tfidf[index]

        if 'train' in self.w2s_path:
            feature_bert_path = '{}/bert_features_train_doc_{}.pth'.format(self.bert_path, str(index))
        elif 'val' in self.w2s_path:
            feature_bert_path = '{}/bert_features_val_doc_{}.pth'.format(self.bert_path, str(index))
        elif 'test' in self.w2s_path:
            feature_bert_path = '{}/bert_features_test_doc_{}.pth'.format(self.bert_path, str(index))

        berts = torch.load(feature_bert_path)
        feature_bert = []
        for sent in berts:
            feature_bert.append(berts[sent])

        feature_bert = feature_bert[:self.doc_max_timesteps]

        G = self.CreateGraph(input_pad, label, w2s_w, feature_bert)


        oracle, generator_inputs = self.features[index]

        context_input_ids = torch.LongTensor(generator_inputs['context_input_ids'])
        context_attention_mask = torch.LongTensor(generator_inputs['context_attention_mask'])
        labels = torch.LongTensor(generator_inputs['labels'])[:config.max_target_len]

        return G, index, oracle, context_input_ids, context_attention_mask, labels

    def __len__(self):
        return self.size


class LoadHiExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.gfiles = [f for f in os.listdir(self.data_root) if f.endswith("graph.bin")]
        logger.info("[INFO] Start loading %s", self.data_root)

    def __getitem__(self, index):
        graph_file = os.path.join(self.data_root, "%d.graph.bin" % index)
        g, label_dict = load_graphs(graph_file)
        # print(graph_file)
        return g[0], index

    def __len__(self):
        return len(self.gfiles)


######################################### Tools #########################################


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res


def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data


def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return:
    '''
    graphs, index,oracle, context_input_ids, context_attention_mask, labels = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph,[index[idx] for idx in sorted_index], [oracle[idx] for idx in sorted_index][0], [context_input_ids[idx] for idx in sorted_index][0], [context_attention_mask[idx] for idx in sorted_index][0], [labels[0][idx] for idx in sorted_index][0]
