import os
import re
import numpy as np
import pandas as pd
import random as rd
from .utils import padding
from .preprocess import Preprocess

columns=["documentId","sentenceId","sentence"]


class PathLineDocuments():
    """Load documents through files.
        Each item corresponds to each document.
        Each sentence in a document is preprocessed in this class.
        
    Args (str): path to the source file or directory
    """
    def __init__(self, source, limit=None):
        self.source = source
        self.limit = limit
        self.num_valid_data=0
        self.is_counted=False
        if os.path.isfile(self.source):
            self.input_files = [self.source]
        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + file for file in self.input_files]
            self.input_files.sort()
        else:
            raise ValueError('input is neither a file nor a path')
    
    def __iter__(self):
        """iterate through the files"""
        for file_name in self.input_files:
            ids, sentences = self.read_tsv(file_name)
            if not self.is_counted:
                self.num_valid_data += len(sentences) - len(np.unique(np.array(ids)[:,1]))*2
            yield (ids, sentences)
        if not self.is_counted:
            print("loaded {} files.".format(len(self.input_files)))
            print("There are {} sentences available for training.".format(self.num_valid_data))
            self.is_counted=True
    
    def read_tsv(self, filepath):
        df = pd.read_csv(filepath, delimiter=',', header=0, names=columns)
        
        document_ids = df["documentId"].values
        sentence_ids = df["sentenceId"].values
        self._ids = np.column_stack((document_ids, sentence_ids))
        self._sentences = df["sentence"].values
        return self._ids, self._sentences
        
class DataLoader():
    def __init__(self, documents, batch_size, n_positive, n_negative, seq_length, token2id, random_seed=42):
        assert isinstance(documents, PathLineDocuments)
        self.documents = documents
        self.batch_size = batch_size
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.seq_length = seq_length
        self.token2id = token2id
        self.unk = token2id['<UNK>']
        self.valid_sen = 0
        self.not_valid_sen = 0
        rd.seed(random_seed)
        
    def __iter__(self):
        tar=[]
        pos=[[] for i in range(self.n_positive)]
        neg=[[] for i in range(self.n_negative)]
        batch_y=np.array(([1.0/self.n_positive]*self.n_positive+[0.0]*self.n_negative)*self.batch_size).reshape(
            self.batch_size, self.n_positive+self.n_negative)
        for ids, document in self.documents:
            if len(document) < 1 + self.n_positive + self.n_negative:
                continue
            ids = np.array(ids)
            sections = np.unique(ids[:,0])
            current_section = ids[0,0]
            for t, s_id in enumerate(ids):
                if current_section != s_id[0]:
                    # new section
                    current_section = s_id[0]
                    self.not_valid_sen += 1
                    continue
                elif t==len(ids)-1:
                    # the end of a document
                    self.not_valid_sen += 1
                    break
                elif current_section != ids[t+1,0]:
                    # the end of a section
                    self.not_valid_sen += 1
                    continue
                elif isinstance(document[t-1], float):
                    if np.isnan(document[t-1]):
                        self.not_valid_sen += 1
                        continue
                elif isinstance(document[t], float):
                    if np.isnan(document[t]):
                        self.not_valid_sen += 1
                        continue
                elif isinstance(document[t+1], float):
                    if np.isnan(document[t+1]):
                        self.not_valid_sen += 1
                        continue
                else:
                    tar.append(self.get_id_sequence(document[t]))
                    pos[0].append(self.get_id_sequence(document[t-1]))
                    pos[1].append(self.get_id_sequence(document[t+1]))
                    for i, n in enumerate(rd.sample(self.other_than(document, t-1, t+1), self.n_negative)):
                        neg[i].append(self.get_id_sequence(n))
                    self.valid_sen += 1
                    if len(tar)==self.batch_size:
                        yield ([np.array(tar)]+[np.array(p) for p in pos]+[np.array(n) for n in neg], batch_y)
                        tar=[]
                        pos=[[] for i in range(self.n_positive)]
                        neg=[[] for i in range(self.n_negative)]
    
    def get_id_sequence(self, line):
        line = list(map(lambda x: self.token2id.get(x, 0), line))
        return padding(line, self.seq_length, self.unk)
    
    def other_than(self, some_list, inf, sup):
        if inf==0:
            return some_list[sup+1:]
        elif sup==len(some_list)-1:
            return some_list[:inf]
        else:
            return some_list[:inf] + some_list[sup+1:]