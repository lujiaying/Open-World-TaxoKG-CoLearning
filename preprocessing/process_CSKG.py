"""
Analysis on WN18RR, CN-100K
Author: Jiaying Lu
Create Date: May 18, 2021
"""


def analysis_CN100k(data_dir: str):
    for fname in ['train100k.txt', 'dev1.txt', 'dev2.txt', 'test.txt']:
        with open('%s/%s' % (data_dir, fname)) as fopen:
            node_set = set()
            edge_cnt = 0
            taxo_edge_cnt = 0
            nontaxo_edge_cnt = 0
            for line in fopen:
                line_list = line.strip().split('\t')
                score = line_list[-1]
                if score == '0':
                    continue
                edge_cnt += 1
                rel = line_list[0]
                if rel == 'IsA':
                    taxo_edge_cnt += 1
                else:
                    nontaxo_edge_cnt += 1
                head, tail = line_list[1], line_list[2]
                node_set.add(head)
                node_set.add(tail)
            print('in %s, #node=%d, #edge=%d, #taxo_e=%d, #nontaxo_e=%d' % (fname, len(node_set),
                  edge_cnt, taxo_edge_cnt, nontaxo_edge_cnt))
            if fname == 'train100k.txt':
                train_node_set = node_set
            if fname == 'test.txt':
                test_node_set = node_set
    train_test_inter = train_node_set.intersection(test_node_set)
    print('#test_train_node intersection=%d, %.4f of #test_node' % (len(train_test_inter),
          len(train_test_inter)/len(test_node_set)))


def analysis_WN18RR(data_dir: str):
    for fname in ['train.txt', 'valid.txt', 'test.txt']:
        with open('%s/%s' % (data_dir, fname)) as fopen:
            node_set = set()
            edge_cnt = 0
            taxo_edge_cnt = 0
            nontaxo_edge_cnt = 0
            for line in fopen:
                line_list = line.strip().split('\t')
                edge_cnt += 1
                rel = line_list[1]
                if rel in ['_hypernym', '_instance_hypernym']:
                    taxo_edge_cnt += 1
                else:
                    nontaxo_edge_cnt += 1
                head, tail = line_list[0], line_list[2]
                node_set.add(head)
                node_set.add(tail)
            print('in %s, #node=%d, #edge=%d, #taxo_e=%d, #nontaxo_e=%d' % (fname, len(node_set),
                  edge_cnt, taxo_edge_cnt, nontaxo_edge_cnt))
            if fname == 'train.txt':
                train_node_set = node_set
            if fname == 'test.txt':
                test_node_set = node_set
    train_test_inter = train_node_set.intersection(test_node_set)
    print('#test_train_node intersection=%d, %.4f of #test_node' % (len(train_test_inter),
          len(train_test_inter)/len(test_node_set)))


if __name__ == '__main__':
    CN100k_dir = 'data/CN-100K'
    # analysis_CN100k(CN100k_dir)

    WN18RR_dir = 'data/WN18RR'
    analysis_WN18RR(WN18RR_dir)
