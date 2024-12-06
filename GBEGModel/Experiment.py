import os

import dgl
import numpy as np
import torch
import nltk
from dataloader.MLMDNews import ExampleSet,graph_collate_fn
from module.HiGraph import  MTSumGraph

import random
from tqdm import tqdm
from config import Config
from utils.utils import (gpu_wrapper, rouge_with_pyrouge)
from torch.utils.data import DataLoader
from transformers import (MBartForConditionalGeneration,
                          MBart50Tokenizer,
                          AdamW)
from module.dynamic_rag import DynamicRagForGeneration
from nltk.tokenize import sent_tokenize, word_tokenize

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = Config()
ROUND = config.ROUND
EPSILON = 1e-10
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)


if config.gpu:
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Experiment(object):

    def __init__(self, load_train=True):

        # Load retriever tokenizer.


        # Load retriever model.
        self.retriever = MTSumGraph(config)
        self.retriever.to("cuda:{}".format(config.select_gpu))

        # Load generator tokenizer.
        self.generator_tokenizer = MBart50Tokenizer.from_pretrained(config.generator_name_or_path,tgt_lang="en_XX")

        # Load generator model.
        self.generator = DynamicRagForGeneration.from_pretrained(config.generator_name_or_path,
                                                                 n_docs=config.top_k,
                                                                 gradient_checkpointing=True)
        #self.generator.to("cuda:0")
        # Load loss.
        self.criterion_cls = torch.nn.CrossEntropyLoss(reduction='none')

        self.modules = ['retriever', 'generator', 'criterion_cls']
        self.DATA_FILE=os.path.join(config.data_dir, "train.label.jsonl")
        self.VALID_FILE = os.path.join(config.data_dir, "val.label.jsonl")
        self.TEST_FILE = os.path.join(config.data_dir, "test.label.jsonl")
        self.FILTER_WORD = os.path.join(config.cache_dir, "filter_word.txt")

        self.train_w2s_path = os.path.join(config.cache_dir, "train.w2s.tfidf.jsonl")
        self.val_w2s_path = os.path.join(config.cache_dir, "val.w2s.tfidf.jsonl")
        self.test_w2s_path = os.path.join(config.cache_dir, "test.w2s.tfidf.jsonl")
        self.bert_path = config.bert_path

        self.train_cached_features_file = os.path.join(config.data_dir, "train_cached_MLMDNews")
        self.val_cached_features_file = os.path.join(config.data_dir, "val_cached_MLMDNews")
        self.test_cached_features_file = os.path.join(config.data_dir, "test_cached_MLMDNews")
        # Load dataset.
        print('----- Loading data -----')
        if load_train:
            self.train_set = ExampleSet('train',self.DATA_FILE,  self.train_cached_features_file, config.doc_max_timesteps, config.sent_max_len, self.FILTER_WORD, self.train_w2s_path, self.bert_path,self.generator_tokenizer )
        self.val_set =ExampleSet('val',self.VALID_FILE,  self.val_cached_features_file, config.doc_max_timesteps, config.sent_max_len, self.FILTER_WORD, self.val_w2s_path, self.bert_path,self.generator_tokenizer )
        self.test_set = ExampleSet('test',self.TEST_FILE,  self.test_cached_features_file, config.doc_max_timesteps, config.sent_max_len, self.FILTER_WORD, self.test_w2s_path, self.bert_path,self.generator_tokenizer )


        for module in self.modules:
            print('--- {}: '.format(module))
            print(getattr(self, module))
            if getattr(self, module) is not None:
                setattr(self, module, gpu_wrapper(getattr(self, module)))

        self.scopes = {'cls': ['retriever'], 'gen': ['generator']}
        for scope in self.scopes.keys():
            setattr(self, scope + '_lr', getattr(config, scope + '_lr'))

        self.iter_num = 0
        self.best_metric = - float('inf')
        self.decay_num = 0
        self.no_improvement = 0

        # Tokenization for BLEU.
        nltk_wordpunk_tokenizer = nltk.tokenize.WordPunctTokenizer()
        self.bleu_tokenizer = lambda x: nltk_wordpunk_tokenizer.tokenize(x)

    def restore_model(self, modules, dirs=None):
        print('Loading the trained best models...')
        if dirs is not None:
            assert len(modules) == len(dirs)
            for module, directory in zip(modules, dirs):
                path = os.path.join(directory, 'best-{}.ckpt'.format(module))
                getattr(self, module).load_state_dict(torch.load(path, map_location=lambda storage, loc: storage),
                                                      strict=True)
        else:
            for module in modules:
                path = os.path.join(config.save_model_dir, 'best-{}.ckpt'.format(module))
                getattr(self, module).load_state_dict(torch.load(path, map_location=lambda storage, loc: storage),
                                                      strict=True)

    def save_step(self, modules):
        for module in modules:
            path = os.path.join(config.save_model_dir, 'best-{}.ckpt'.format(module))
            torch.save(getattr(self, module).state_dict(), path)
        print('Saved model checkpoints into {}...\n\n\n\n\n\n\n\n\n\n\n\n'.format(config.save_model_dir))

    def zero_grad(self):
        for scope in self.scopes:
            getattr(self, scope + '_optim').zero_grad()

    def step(self, scopes):
        if config.max_grad_norm is not None:
            grouped_params = []
            for scope in scopes:
                grouped_params.extend(getattr(self, scope + '_grouped_parameters'))

            clip_grad_norm_(grouped_params, config.max_grad_norm)

        for scope in scopes:
            # Optimize.
            getattr(self, scope + '_optim').step()

    def update_lr_by_half(self):
        self.decay_num += 1
        for scope in self.scopes:
            setattr(self, scope + '_lr', getattr(self, scope + '_lr') / 2)  # Half the learning rate.
            for param_group in getattr(self, scope + '_optim').param_groups:
                param_group['lr'] = getattr(self, scope + '_lr')
            print('{}: {}'.format(scope + '_lr', getattr(self, scope + '_lr')))

    def set_requires_grad(self, modules, requires_grad):
        if not isinstance(modules, list):
            modules = [modules]
        for module in modules:
            for param in getattr(self, module).parameters():
                param.requires_grad = requires_grad

    def set_training(self, mode):
        for module in self.modules:
            if getattr(self, module) is not None:
                getattr(self, module).train(mode=mode)

    def train(self):
            
        self.build_optim()

        # Train.
        epoch = 0
        self.zero_grad()
        while True:
            self.train_epoch(epoch)
            epoch += 1
            if self.decay_num >= config.max_decay_num:
                break

        # Test.
        self.test()

    def build_optim(self):
        # Set trainable parameters, according to the frozen parameter list.
        for scope in self.scopes.keys():
            optimizer_grouped_parameters = [
                {'params': [],
                 'weight_decay': config.weight_decay},
                {'params': [],
                 'weight_decay': 0.0},
            ]
            no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

            for module in self.scopes[scope]:
                if getattr(self, module) is not None:
                    for n, p in getattr(self, module).named_parameters():
                        # k is the parameter name; v is the parameter value.
                        if p.requires_grad:
                            # Weight decay.
                            if not any(nd in n for nd in no_decay):
                                print("[{} Trainable:]".format(module), n)
                                optimizer_grouped_parameters[0]['params'].append(p)
                            else:
                                print("[{} Trainable (bias/LN):]".format(module), n)
                                optimizer_grouped_parameters[1]['params'].append(p)
                        else:
                            print("[{} Frozen:]".format(module), n)

            if config.optimizer == 'adam':
                setattr(self, scope + '_optim', AdamW(optimizer_grouped_parameters, lr=getattr(self, scope + '_lr')))
            else:
                raise ValueError()

            setattr(self,
                    scope + '_grouped_parameters',
                    optimizer_grouped_parameters[0]['params'] + optimizer_grouped_parameters[1]['params'])

    def test(self):
        self.restore_model(['retriever', 'generator'])
        # Evaluate.
        beam_size = 5
        self.seq_evaluate_gen(test=True, beam_size=beam_size)
        
    def train_epoch(self, epoch_id):

        train_dataloader = DataLoader(self.train_set,
                                      batch_size=config.train_batch_size // config.gradient_accumulation_steps,
                                      shuffle=True,
                                      num_workers=config.num_workers,
                                      collate_fn=graph_collate_fn)

        for data in train_dataloader:
            self.iter_num += 1
            self.set_training(mode=True)

            if config.target_task in ['MLMDNews'
                                      ]:
                #Process data.
                #data = self.cuda_data(*data)

                G, index, oracle, context_input_ids, context_attention_mask, labels = data

                G = G.to(torch.device("cuda:{}".format(config.select_gpu)))
                context_input_ids = context_input_ids.unsqueeze(0).to(torch.device("cuda:{}".format(config.select_gpu)))
                context_attention_mask = context_attention_mask.unsqueeze(0).to(torch.device("cuda:{}".format(config.select_gpu)))
                labels = labels.unsqueeze(0).to(torch.device("cuda:{}".format(config.select_gpu)))
                # Forward.
                retriever_outputs = self.retriever.forward(G)
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                label = G.ndata["label"][snode_id].sum(-1)  # [n_nodes]
                G.nodes[snode_id].data["loss"] = self.criterion_cls(retriever_outputs, label).unsqueeze(-1)  # [n_nodes, 1]
                loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
                ret_loss = loss.mean()

                G.nodes[snode_id].data["p"] = retriever_outputs
                g = dgl.unbatch(G)[0]
                snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                N = len(snode_id)
                p_sent = g.ndata["p"][snode_id]
                p_sent = p_sent.view(-1, 2)  # [node, 2]
                # Generation loss.
                if config.loss_alpha != 0 :
                    doc_scores, retriever_topk_indices = torch.topk(p_sent[:, 1], min(config.top_k, N))
                    doc_scores = doc_scores.unsqueeze(0)
                    retriever_topk_indices = retriever_topk_indices.cpu().tolist()

                    if len(retriever_topk_indices) < config.top_k:
                        doc_scores = torch.cat([doc_scores, gpu_wrapper(
                             torch.zeros((1, config.top_k - len(retriever_topk_indices)))).fill_(-float('inf'))], dim=1)
                        pad_count = config.top_k - len(retriever_topk_indices)
                        if pad_count > len(retriever_topk_indices):
                            retriever_topk_indices = retriever_topk_indices + retriever_topk_indices + [retriever_topk_indices[0]] * (pad_count - len(retriever_topk_indices))
                        else:
                            retriever_topk_indices = retriever_topk_indices + [i for i in range(0, pad_count)]
                        print(retriever_topk_indices)
                    # import pdb
                    # pdb.set_trace()
                    generator_outputs = self.generator(context_input_ids=context_input_ids[:, retriever_topk_indices].contiguous().view(context_input_ids.shape[0] * config.top_k, -1),
                                                           context_attention_mask=context_attention_mask[:, retriever_topk_indices].contiguous().view(context_attention_mask.shape[0] * config.top_k, -1),
                                                           doc_scores=doc_scores,
                                                           labels=labels)
                    seq_loss = generator_outputs.loss
                    consistency_loss = generator_outputs.consistency_loss

                else:
                    seq_loss = 0

                tot_loss = seq_loss * config.loss_alpha + ret_loss

                tot_loss = tot_loss + config.consistency_alpha * consistency_loss

                # Backward.
                if config.gradient_accumulation_steps > 1:
                    tot_loss = tot_loss / config.gradient_accumulation_steps

                # print(tot_loss)
                tot_loss.backward()

                if self.iter_num % config.gradient_accumulation_steps == 0:
                    # ----- Backward for scopes: ['cls', 'gen'] -----
                    self.step(['cls', 'gen'])

                    self.zero_grad()
            else:
                raise ValueError()

            # Evaluation.
            if self.iter_num % (config.save_steps * config.gradient_accumulation_steps) == 0:
                beam_size = 5    
                no_improvement = self.seq_evaluate_gen(test=True, beam_size=beam_size)

                # Learning rate decay.
                if no_improvement and self.iter_num > config.start_decay:
                    self.no_improvement += 1
                else:
                    self.no_improvement = 0

                if self.no_improvement == config.no_improvement_decay:
                    self.update_lr_by_half()
                    self.no_improvement = 0

    def seq_evaluate_gen(self, test, beam_size):
        self.set_training(mode=False)

        print('beam_size = {}'.format(beam_size))

        if test:
            the_set = self.test_set
        else:
            the_set = self.val_set
        eval_dataloader = DataLoader(the_set, batch_size=1, shuffle=False, num_workers=config.num_workers,collate_fn=graph_collate_fn)

        # Eval!
        print("\n\n\n\n***** Running evaluation *****")
        print("  Num examples = {}".format(len(the_set)))
        print("  Batch size = {}".format(1))

        predictions = []
        topks = []
        doc_scoreses = []

        top_5 = [False, True][0]
        tot = 0

        for data in tqdm(eval_dataloader):
            tot += 1
            if tot > 3 and top_5:
                break
            # Process data.
            G, index, oracle, context_input_ids, context_attention_mask, labels = data
            G = G.to(torch.device("cuda:{}".format(config.select_gpu)))
            context_input_ids = context_input_ids.unsqueeze(0).to(torch.device("cuda:{}".format(config.select_gpu)))
            context_attention_mask = context_attention_mask.unsqueeze(0).to(torch.device("cuda:{}".format(config.select_gpu)))

            with torch.no_grad():
                retriever_outputs = self.retriever.forward(G)
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                label = G.ndata["label"][snode_id].sum(-1)  # [n_nodes]
                G.nodes[snode_id].data["loss"] = self.criterion_cls(retriever_outputs, label).unsqueeze(
                    -1)  # [n_nodes, 1]
                G.nodes[snode_id].data["p"] = retriever_outputs
                g = dgl.unbatch(G)[0]
                snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                N = len(snode_id)
                p_sent = g.ndata["p"][snode_id]
                p_sent = p_sent.view(-1, 2)

                doc_scores, retriever_topk_indices = torch.topk(p_sent[:, 1], min(config.top_k, N))
                doc_scores = doc_scores.to("cuda:{}".format(config.select_gpu))
                doc_scores = doc_scores.unsqueeze(0)
                retriever_topk_indices = retriever_topk_indices.cpu().tolist()
                
                if len(retriever_topk_indices) < config.top_k:
                    doc_scores = torch.cat([doc_scores, gpu_wrapper(
                            torch.zeros((1, config.top_k - len(retriever_topk_indices)))).fill_(-float('inf'))], dim=1)
                    pad_count = config.top_k - len(retriever_topk_indices)
                    if pad_count > len(retriever_topk_indices):
                        retriever_topk_indices = retriever_topk_indices + retriever_topk_indices + [retriever_topk_indices[0]] * (pad_count - len(retriever_topk_indices))
                    else:
                        retriever_topk_indices = retriever_topk_indices + [i for i in range(0, pad_count)]
                    print(retriever_topk_indices)

                if config.loss_alpha != 0:
                    outputs = self.generator.generate(context_input_ids=context_input_ids[:, retriever_topk_indices].contiguous().view(context_input_ids.shape[0] * config.top_k, -1),
                                                      context_attention_mask=context_attention_mask[:, retriever_topk_indices].contiguous().view(context_attention_mask.shape[0] * config.top_k, -1),
                                                      doc_scores=doc_scores,
                                                      num_beams=beam_size,
                                                      min_length=config.min_length,
                                                      max_length=config.max_target_len,
                                                      no_repeat_ngram_size=config.no_repeat_ngram_size,
                                                      length_penalty=config.length_penalty,
                                                      )
                    assert isinstance(outputs, torch.Tensor)
                    assert outputs.shape[0] == 1

                    # Predictions:
                    decoded_pred = self.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True,lang="en_XX")

                    cleaned_prediction = ["\n".join(sent_tokenize(" ".join(word_tokenize(pred)))) for pred in decoded_pred]
                else:
                    cleaned_prediction = [" prediction because loss_alpha = 0."]
                predictions.extend(cleaned_prediction)

                # top_k:
                decoded_topk = self.generator_tokenizer.batch_decode(context_input_ids[:, retriever_topk_indices].contiguous().view(config.top_k, -1), skip_special_tokens=True,lang="en_XX")
                cleaned_topk = "\n".join(sent_tokenize(" ".join(word_tokenize(" ".join([sent for sent, prob in zip(decoded_topk, torch.softmax(doc_scores[0], dim=0)) if prob > 1e-10])))))
                topks.append(cleaned_topk)
                doc_scoreses.append(doc_scores[0])


        # Load references.
        references = ["\n".join(sent_tokenize(" ".join(word_tokenize(sent)))) for sent in the_set.get_references()]

        # ROUGE.
        rouge1, rouge2, rougeL = rouge_with_pyrouge(preds=predictions, refs=references)
        print(rouge1, rouge2, rougeL)

        rouge1_topk, rouge2_topk, rougeL_topk = rouge_with_pyrouge(preds=topks, refs=references)

        print(rouge1_topk, rouge2_topk, rougeL_topk)

        if config.loss_alpha != 0:
            metric = rouge1 + rouge2 + rougeL
        else:
            metric = rouge1_topk + rouge2_topk + rougeL_topk

        if not test and metric > self.best_metric:
            self.best_metric = metric
            self.save_step(['retriever', 'generator'])

            peep_num = 3
            for sent_id in range(peep_num):
                print('Pred:\n{}'.format(predictions[sent_id]))
                print('-' * 20)
                print('topk:\n{}'.format(topks[sent_id]))
                print('-' * 20)
                print('Ref:\n{}'.format(references[sent_id]))
                print('-' * 20)
                print()
                print('=' * 50)

        self.set_training(mode=True)

        base_name = '{}.gen'.format('test' if test else 'valid')
        save_path = os.path.join(config.sample_dir, base_name)
        torch.save((predictions, references), save_path)

    def number_parameters(self):
        print('Number of retriever parameters', sum(p.numel() for p in self.retriever.parameters()))
        print('Number of generator parameters', sum(p.numel() for p in self.generator.parameters()))

    @staticmethod
    def cuda_data(*data, **kwargs):
        if len(data) == 0:
            raise ValueError()
        elif len(data) == 1:
            return gpu_wrapper(data[0], **kwargs)
        else:
            return [gpu_wrapper(item, **kwargs) for item in data[3:5]]


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    # print(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


