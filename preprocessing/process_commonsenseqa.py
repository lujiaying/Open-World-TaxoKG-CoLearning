""" 
Get concepts in commonsenseqa
Author: Anonymous Siamese
Create Date: May 12, 2021
"""


import json
import tqdm


def extract_concepts_in_answer(data_dir: str, output_path: str):
    concept_res = {}
    for name in ['train_rand_split.jsonl', 'dev_rand_split.jsonl', 'test_rand_split_no_answers.jsonl']:
        with open(data_dir+'/'+name) as fopen:
            for line in tqdm.tqdm(fopen):
                line_res = json.loads(line.strip())
                concept = line_res['question']['question_concept']
                if concept not in concept_res:
                    concept_res[concept] = 0
                concept_res[concept] += 1
                for _ in line_res['question']['choices']:
                    concept = _['text']
                    if concept not in concept_res:
                        concept_res[concept] = 0
                    concept_res[concept] += 1
    concept_res = sorted(concept_res.items(), key=lambda _: _[1], reverse=True)
    with open(output_path, 'w') as fwrite:
        for surface, freq in concept_res:
            fwrite.write('%s\t%s\n' % (surface, freq))


if __name__ == '__main__':
    commonsenseqa_dir = 'data/commonsenseqa'
    concept_freq_path = '%s/concept_freq.txt' % (commonsenseqa_dir)
    extract_concepts_in_answer(commonsenseqa_dir, concept_freq_path)
