from datasetReader import DatasetReader
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd

class QLDatasetReader(DatasetReader):

    def words(self, data_path, tokenizer, exclude_fn=lambda x: False):
        for line in open(data_path):
            data = line.split("\t")
            if len(data) < 8 or data[6] is None or data[7] is None:
                continue
            if exclude_fn(data):
                continue
            for w in tokenizer(data[6]) + tokenizer(data[7]):
                yield w

    def conversations(self, data_path, exclude_fn=lambda x: False, yield_fn=lambda x: (x[6], x[7])):
        for line in open(data_path):
            data = line.split("\t")
            if len(data) < 8 or data[6] is None or data[7] is None:
                continue
            if exclude_fn(data):
                continue
            yield yield_fn(data)

class FilteredDatasetReader(QLDatasetReader):

    def __init__(self):
        df = pd.read_csv("joined-ids-predictions-from_test.csv", sep="\t")
        only_good = (df['prediction'] > 0) & (df['from_test'] == False)
        self.allowed_ids = set(df[only_good]['question_id'])
        print("allowed ids", len(self.allowed_ids))

    def exclude_fn(self, item):
        return item[0] not in self.allowed_ids

    def good_ids(data_path):
        df_list = []
        data_path_pred = os.path.join(data_path, 'predictions')
        tree = ET.parse(os.path.join(data_path_pred, 'exclude-2016.xml'))
        root = tree.getroot()
        test_ids = set()
        for thread in root:
            qid = thread.find('Thread').find('RelQuestion').attrib['RELQ_ID']
            test_ids.add(qid)
        print("excluded 2016")
        tree = ET.parse(os.path.join(data_path_pred, 'exclude-2017.xml'))
        root = tree.getroot()
        for thread in root:
            qid = thread.find('Thread').find('RelQuestion').attrib['RELQ_ID_ORIG']
            test_ids.add(qid)
        print("excluded 2017")
        tree = ET.parse(os.path.join(data_path_pred,
                                     "reformatted_prod-qatarliving_2-qatarlividb14861-2016-02-23.dump.with-orig.xml"))
        ds = defaultdict(lambda: defaultdict(lambda:{}))
        root = tree.getroot()
        for thread in root:
            question = thread.find('RelQuestion')
            for c in thread.findall('RelComment'):
                qid = question.attrib['RELQ_ID_ORIG']
                cid = c.attrib['RELC_ID_ORIG']
                entry = {}
                entry['orig_question_id'] = qid
                entry['orig_comment_id'] = cid
                entry['comment_id'] = c.attrib['RELC_ID']
                entry['question_id'] = question.attrib['RELQ_ID']
                entry['date_question'] = question.attrib['RELQ_DATE']
                entry['date_comment'] = c.attrib['RELC_DATE']
                entry['from_test'] = qid in test_ids
                ds[qid][cid] = entry
                df_list.append(entry)
        pred_files = [f for f in os.listdir(data_path_pred) if re.match("semeval.dump\d+.txt.pred", f)]
        for prediction_file in pred_files:
            id_file = prediction_file.replace("pred","test.ids")
            with open(os.path.join(data_path_pred, prediction_file)) as pfile, open(os.path.join(data_path_pred, id_file)) as ifile:
                for prediction, ids in zip(pfile, ifile):
                    qid, cid = ids.strip().split("\t")
                    ds[qid][cid]['prediction'] = float(prediction.strip())
        column_order = ["question_id", "orig_question_id", "comment_id", "orig_comment_id", 'date_question',
                        'date_comment', 'from_test', 'prediction']
        with open('joined-ids-predictions-from_test.csv', 'w') as out:
            out.write("\t".join(column_order)+ "\n")
            for qid in ds:
                for cid in ds[qid]:
                    cur = ds[qid][cid]
                    out.write("\t".join(str(cur[column]) for column in column_order) + "\n")


    def words(self, data_path, tokenizer):
        return super(FilteredDatasetReader, self).words(data_path, tokenizer, self.exclude_fn)

    def conversations(self, data_path, yield_fn= lambda x: x):
        return super(FilteredDatasetReader, self).conversations(data_path, self.exclude_fn, yield_fn)

#r = FilteredDatasetReader()
#with open('/home/martin/data/qatarliving/matchedPairs_ver5/matchPairs_B/train-filtered-feb.csv', 'w') as out:
#    for conv in r.conversations("/home/martin/data/qatarliving/matchedPairs_ver5/matchPairs_B/train.csv", None, yield_fn = lambda x: x):
#        out.write("\t".join(conv).strip() + "\n")


#FilteredDatasetReader.good_ids("/home/martin/data/qatarliving/matchedPairs_ver5/matchPairs_B")