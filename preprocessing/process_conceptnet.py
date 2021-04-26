"""
Derive taxonomy from conceptnet
Author: Jiaying Lu
Create Date: Apr 25, 2021
"""

import tqdm


def extract_edges_contains_concept(data_path: str, concept: str, out_path: str):
    node = '/c/en/%s' % (concept)
    with open(data_path) as fopen, open(out_path, 'w') as fwrite:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            head = line_list[2]
            tail = line_list[3]
            if head == node or tail == node:
                fwrite.write(line)


def extract_edges_in_English(data_path: str, out_path: str):
    with open(data_path) as fopen, open(out_path, 'w') as fwrite:
        for line in tqdm.tqdm(fopen):
            line_list = line.strip().split('\t')
            head = line_list[2]
            tail = line_list[3]
            if head.startswith('/c/en/') and tail.startswith('/c/en/'):
                fwrite.write(line)


if __name__ == "__main__":
    data_path = 'data/conceptnet/conceptnet-assertions-5.7.0.csv'
    out_path = 'data/conceptnet/conceptnet-assertions-5.7.0.en.csv'
    # extract_edges_in_English(data_path, out_path)

    data_path = 'data/conceptnet/conceptnet-assertions-5.7.0.en.csv'
    concept = 'artificial_intelligence'
    out_path = 'data/conceptnet/edges_of_concept/%s.csv' % (concept)
    extract_edges_contains_concept(data_path, concept, out_path)
