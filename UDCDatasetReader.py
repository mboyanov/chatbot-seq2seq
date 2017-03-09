import unicodecsv as csv
import udcparser
from datasetReader import DatasetReader


class UDCDatasetReader(DatasetReader):

    def __init__(self, training = True):
        self.training = training

    def words(self, data_path, tokenizer):
        counter = 0
        r = csv.reader(open(data_path, 'rb'), encoding='utf-8')
        next(r)
        for conversation in r:
            if conversation[2] != '1':
                continue
            utterances = udcparser.parse(conversation)
            if utterances is not None:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokenized_utterances = [tokenizer(utterance) for utterance in utterances]
                for tokens in tokenized_utterances:
                    for w in tokens:
                        yield w

    def conversations(self, data_path):
        counter = 0
        r = csv.reader(open(data_path, 'rb'), encoding='utf-8')
        next(r)
        for conversation in r:
            if self.training and conversation[2] != '1':
                continue
            utterances = udcparser.parse(conversation)
            if utterances is not None:
                counter += 1
                if counter % 100000 == 0:
                    print("  tokenizing line %d" % counter)
                utterances = [ut for ut in utterances if len(ut) > 0]
                for i in range(len(utterances) - 1):
                    ut1 = utterances[i]
                    ut2 = utterances[i + 1]
                    yield(ut1, ut2)
