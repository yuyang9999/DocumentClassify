# encoding: utf-8

import numpy as np
import sys

class Bayes:
    def __init__(self, word_set_path):
        #load the word set path
        word_list = []
        word_index_dic = {}
        with open(word_set_path) as f:
            for l in f:
                l = l.replace('\n', '')
                if l != '':
                    word_list.append(l)
                    word_index_dic[l] = len(word_list) - 1

        self.word_list = np.array(word_list)
        self.word_index_dic = word_index_dic


    def convert_line_to_vector(self, doc):
        np_line = np.zeros(len(self.word_list))
        for part in doc:
            if not part in self.word_index_dic:
                continue
            np_line[self.word_index_dic[part]] += 1
        return np_line


    def train(self, doc_matrix, labels, label_ids, output_path):
        total_doc_cnt = len(doc_matrix)
        label_distri = np.zeros(len(label_ids))
        for lab in labels:
            label_distri[label_ids.index(lab)] += 1

        for idx in range(len(label_distri)):
            label_distri[idx] = np.log(label_distri[idx] / total_doc_cnt)

        label_word_cnt = np.zeros(len(label_ids))
        for idx in range(len(label_word_cnt)):
            label_word_cnt[idx] = 2 #平滑系数, 用2还是啥？

        #平滑化，使每个label 默认次数为1
        label_words = np.ones((len(label_ids), len(self.word_list)))

        for idx in range(len(doc_matrix)):
            doc = doc_matrix[idx]
            label_idx = label_ids.index(labels[idx])
            label_word_cnt[label_idx] += len(doc)
            doc_vector = self.convert_line_to_vector(doc)
            label_words[label_idx] += doc_vector

        for idx in range(len(label_ids)):
            label_words[idx] = np.log(label_words[idx] / label_word_cnt[idx])

        #save the model
        with open(output_path, 'w') as f:
            #write the ids
            f.write(','.join(label_ids) + '\n')
            #write the label prob distribution
            f.write(','.join([str(f) for f in label_distri]) + '\n')
            #write the word vector distribution for each label
            for vector in label_words:
                f.write(','.join([str(f) for f in vector]) + '\n')


    def load_model(self, model_path):
        label_ids = []
        label_distri = []
        label_word_dis = []

        with open(model_path) as f:
            label_ids.extend(f.readline().replace('\n', '').split(','))
            label_distri.extend([float(v) for v in f.readline().replace('\n', '').split(',')])
            for idx in range(len(label_ids)):
                word_dis = [float(v) for v in  f.readline().replace('\n', '').split(',')]
                label_word_dis.append(np.array(word_dis))
        return label_ids, label_distri, label_word_dis


    def predict(self, model_path, doc_matrix):
        label_ids, label_distri, label_word_dis = self.load_model(model_path)
        predict_labels = []
        for doc in doc_matrix:
            doc_vector = self.convert_line_to_vector(doc)
            max_idx = 0
            max_sum = -sys.float_info.max
            for idx in range(len(label_distri)):
                tmp_sum = np.sum(doc_vector * label_word_dis[idx]) + label_distri[idx]
                if tmp_sum > max_sum:
                    max_sum = tmp_sum
                    max_idx = idx
            predict_labels.append(label_ids[max_idx])
        return predict_labels