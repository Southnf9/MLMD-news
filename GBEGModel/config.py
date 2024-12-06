import os
import torch
import glob


class Config(object):

    def __init__(self):

        self.target_task = 'MLMDNews'


        self.generator = ['dynamic-rag',
                          ][0] 
        self.generator_name_or_path = {'dynamic-rag': '/root/mbart-many-to-one',
                                       }[self.generator]
        #MTGNN-SUM
        self.data_dir = './data/MLMDNews'
        self.cache_dir = './cache/MLMDNews'

        self.DATA_FILE = os.path.join(self.data_dir, "train.label.jsonl")
        self.VALID_FILE = os.path.join(self.data_dir, "val.label.jsonl")
        self.VOCAL_FILE = os.path.join(self.cache_dir, "vocab")
        self.FILTER_WORD = os.path.join(self.cache_dir, "filter_word.txt")

        self.restore_model = None
        self.bert_path = './bert_features_MLMDNews'
        self.word_embedding =True
        self.vocab_size = 50000
        self.word_emb_dim = 768
        self.embed_train = False
        self.n_iter = 1
        self.feat_embed_size = 50
        self.n_layers = 1
        self.lstm_hidden_state = 128
        self.lstm_layers = 2
        self.bidirectional = True
        self.n_feature_size = 128
        self.hidden_size = 64
        self.n_head = 8
        self.recurrent_dropout_prob = 0.1
        self.atten_dropout_prob = 0.1
        self.ffn_dropout_prob = 0.1
        self.use_orthnormal_init = True
        self.sent_max_len = 100
        self.doc_max_timesteps = 50
        self.lr_descent = True
        self.grad_clip = True
        self.max_grad_norm = 1.0
        self.m = 5
        self.ffn_inner_hidden_size = 512
        self.cuda = True
        self.select_gpu = 0


        # Training configuration.
        self.max_grad_norm = 1.0
        self.cls_lr = 5e-6
        self.gen_lr = 5e-5 

        self.overwrite_cache = False 
        self.weight_decay = 0.0  

        self.start_decay = 0
        self.max_decay_num = 3
        self.no_improvement_decay = 5
        self.optimizer = 'adam'
        self.filtered_oracle = False
        
        self.early_preprocess = True

        self.train_batch_size = 8 
        self.eval_batch_size = 1
        self.test_batch_size = 1
        self.gradient_accumulation_steps = 8 
        assert self.train_batch_size % self.gradient_accumulation_steps == 0

        # Miscellaneous.
        self.num_workers = 8 
        self.ROUND = 4
        self.seed = [0, 1, 2, 3, 4][0] 
        self.gpu = torch.cuda.is_available()

        # Method-related.
        self.use_oracle = True
        self.use_query = False

        self.oracle_type = ['greedy', ][0]
        self.oracle_train = [False, True][1]
        if self.oracle_train:
            self.hybrid_train = [False, True][1]
        self.oracle_test = [False, True][0]
        self.loss_alpha = [0, 0.1, 0.5, 1, 5][1]
        self.window_size = 0
        self.top_k = 10
        self.min_length = 350
        self.no_repeat_ngram_size = 3

        self.max_source_len = 128
        self.max_target_len = 500

        self.consistency_alpha = [0, 1, 2, 3, 5, 10, 15][0]
        self.detach_generator_consistency = [False, True][1]
        self.length_penalty = 1
        #self.save_steps = 500
        self.save_steps = 300
        self.retriever_save_steps = 600
            


        # Directories.
        self.log_dir = './outputs/logs'
        #remove_all_under(self.log_dir)
        self.save_model_dir = './outputs/saved_model'
        self.sample_dir = './outputs/sampled_results'
        self.tmp_dir = './outputs/temp_results'

    def model_specific_dir(self, root):
        """ model-normalization """
        directory = 'MLMDNews'

        ret = os.path.join(root, directory)
        if not os.path.exists(ret):
            os.mkdir(ret)
        return ret


def remove_all_under(directory):
    for file in glob.glob(os.path.join(directory, '*')):
        os.remove(file)
