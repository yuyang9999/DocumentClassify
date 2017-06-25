
import Bayes
import numpy as np

def post_analysis(predict_labels, expect_labels):
    assert(len(predict_labels) == len(expect_labels))
    arr1 = np.array(predict_labels)
    arr2 = np.array(expect_labels)
    correct_cnt = sum(arr1 == arr2)
    precision = correct_cnt / len(predict_labels)
    print('precision is ' + str(precision))


def test():
    labels = []
    label_ids = set()
    doc_matrix = []

    dir = 'data/'
    train_file = 'train.txt'
    test_file = 'test.txt'
    word_set_file = 'all_words.txt'
    model_file = 'model.txt'

    with open(dir + train_file) as f:
        for l in f:
            l = l.replace('\n','')
            if l == '':
                continue
            comps = l.split('\t')
            assert(len(comps) == 2)
            if comps[1] == '':
                continue
            labels.append(comps[0])
            doc_matrix.append(comps[1].split(','))
            label_ids.add(comps[0])
    bayes_model = Bayes.Bayes(dir + word_set_file)
    bayes_model.train(doc_matrix, labels, list(label_ids), dir + model_file)

    #open the test file
    expect_labels = []
    predict_docs = []
    with open(dir + test_file) as f:
        for l in f:
            l = l.replace('\n', '')
            if l == '':
                continue
            comps = l.split('\t')
            if comps[1] == '':
                continue
            assert(len(comps) == 2)
            expect_labels.append(comps[0])
            predict_docs.append(comps[1].split(','))
    predict_labels = bayes_model.predict(dir + model_file, predict_docs)
    post_analysis(predict_labels, expect_labels)


if __name__ == '__main__':
    test()