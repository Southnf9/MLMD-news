import argparse
import re, json
import nltk.data
import py3langid as langid
from utils.utils import _get_word_ngrams
from nltk import  word_tokenize
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# 加载多语言BERT模型和tokenizer
model_name = '/root/bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


sent_detector_en = nltk.data.load('tokenizers/punkt/english.pickle')
sent_detector_de = nltk.data.load('tokenizers/punkt/german.pickle')
sent_detector_fr = nltk.data.load('tokenizers/punkt/french.pickle')
sent_detector_es = nltk.data.load('tokenizers/punkt/spanish.pickle')

sent_detector_dict={
    'fr':sent_detector_fr,
    'de':sent_detector_de,
    'es':sent_detector_es,
    'en':sent_detector_en
}
lanauge_dict={
    'fr':'french',
    'de':'german',
    'es':'spanish',
    'en':'english'
}
sent_limit = 64
def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]


            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def greedy_selection_ByMbert(doc_sent_list, abstract_sent_list, summary_size):

    abstract_embedding =  get_embeddings([" ".join(abstract_sent_list)])
    all_doc_sent_embedding =get_embeddings(doc_sent_list)
    all_similarity = calculate_similarity_matrix(all_doc_sent_embedding,abstract_embedding)
    top_10_indices = np.argsort(all_similarity.reshape(all_similarity.shape[0]))[-summary_size:][::-1]
    return  sorted(top_10_indices.tolist())





def format_to_lines(input_path, output_path, task):

    doc_path = '{}/{}/{}.src.txt'.format(input_path, task,task)
    summary_path = '{}/{}/{}.tgt.txt'.format(input_path, task,task)
    lanauge_path = '{}/{}/{}_lanauge.txt'.format(input_path, task,task)
    save_path = '{}/{}.label.jsonl'.format(output_path, task)

    fout = open(save_path, 'w')
    f_doc = open(doc_path,"r",encoding='utf-8')
    f_summary = open(summary_path, "r", encoding='utf-8')
    f_lanauge = open(lanauge_path, "r", encoding='utf-8')

    documents = f_doc.readlines()
    summarys = f_summary.readlines()[0:len(documents)]
    lanauges = eval(f_lanauge.read())

    index = 0
    tag=' story_separator_special_tag '
    for doc,sum,l in zip(documents,summarys,lanauges):

        tmp_id = "{}_{}".format(task, index)
        dataset = {}
        sent_start_index = 0
        doc_list = doc.split(tag)
        doc_sent_list = []
        lanauge_key = [i[0] for i in l]
        lanauges_dict = dict(l)

        lanauge_tag = []
        for doc_index, doc in enumerate(doc_list):

            if doc_index in lanauge_key:
                lanauge = lanauges_dict[doc_index]
            else:
                lanauge = 'en'
            sent_list = sent_detector_dict[lanauge].tokenize(doc.strip())
            sent_list = process_article(sent_list, lanauge_dict[lanauge])
            doc_sent_list += sent_list

            sent_end_index = sent_start_index + len(sent_list)
            lanauge_tag += [{i:lanauge} for i in range(sent_start_index,sent_end_index)]
            # tmp_tag[lanauge].append([sent_start_index,sent_end_index])
            sent_start_index = sent_end_index

        sum_sent_list =sent_detector_en.tokenize(sum.strip())

        # doc_sent_list = process_article(doc_sent_list, lanauge_dict[lanauge])

        doc_sent_list_split = [e.strip().split() for e in doc_sent_list]
        sum_sent_list_split = [e.strip().split() for e in sum_sent_list]

        # sent_labels = greedy_selection(doc_sent_list_split, sum_sent_list_split, 10)
        sent_labels = greedy_selection_ByMbert(doc_sent_list, sum_sent_list, 10)
        dataset["id"] = tmp_id
        dataset["text"] = doc_sent_list
        dataset["summary"] = sum_sent_list
        dataset["label"] = sorted(sent_labels)
        dataset["lanauge"] = lanauge_tag
        fout.write(json.dumps(dataset) + '\n')
        index += 1

    fout.close()
    f_doc.close()
    f_summary.close()

def insert_new( article_list, sent, lanauge):

    token_list = word_tokenize(sent,language = lanauge)

    while len(token_list) > sent_limit:
        article_list.append(" ".join(token_list[:sent_limit]))
        token_list = token_list[sent_limit:]
    article_list.append(" ".join(token_list))

    return article_list

def process_article(article,lanauge):

        new_article = []

        for sent in article:
            new_article =  insert_new(new_article, sent,lanauge)

        return new_article

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取最后一个隐藏层的平均值作为句子嵌入
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# 计算句子列表的嵌入表示
def get_embeddings(sent_list):
    return np.vstack([get_sentence_embedding(sent) for sent in sent_list])

def calculate_similarity_matrix(doc_embeddings, abstract_embeddings):
    return cosine_similarity(doc_embeddings, abstract_embeddings)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing dataset')

    parser.add_argument('--input_path', type=str, default='./data/MLMDNews_dataset', help='The dataset directory.')
    parser.add_argument('--output_path', type=str, default='data/MLMDNews', help='The dataset directory.')
    parser.add_argument('--task', type=str, default='train', help='dataset [train|val|test]')

    args = parser.parse_args()

    format_to_lines(args.input_path, args.output_path, args.task)
