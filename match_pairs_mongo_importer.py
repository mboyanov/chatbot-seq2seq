from QLDatasetReader import QLDatasetReader

import os
from db import db

reader = QLDatasetReader()
dataset_home = "/home/martin/data/qatarliving/matchedPairs_ver5/matchPairs_SB"
datasets = {
    "dev": os.path.join(dataset_home, "SemEval2016_dev_taskA.xml_MatchPairs_SB.txt"),
    "test": os.path.join(dataset_home, "SemEval2016_test_taskA.xml_MatchPairs_SB.txt"),
    "train": os.path.join(dataset_home, "SemEval2016_train_part1part2_taskA.xml_MatchPairs_SB.txt")
}


def flat2dict(ex, dataset):
    return {
        "question_id": ex[0],
        'comment_id': ex[1],
        'relevance': float(ex[4]),
        'question_sentence': ex[6],
        'comment_sentence': ex[7],
        'entire_question': ex[8],
        'entice_comment': ex[9],
        'dataset': dataset
    }

def main():
    for dataset, file_path in datasets.items():
        for qa_pair in reader.conversations(file_path, yield_fn= lambda x:x):
            d = flat2dict(qa_pair, dataset)
            db.matched_pairs.insert(d)


main()