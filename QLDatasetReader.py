from datasetReader import DatasetReader


class QLDatasetReader(DatasetReader):

    def words(self, data_path, tokenizer):
        for line in open(data_path):
            data = line.split("\t")
            if len(data) < 8 or data[6] is None or data[7] is None:
                continue
            for w in tokenizer(data[6]) + tokenizer(data[7]):
                yield w

    def conversations(self, data_path, tokenizer):
        for line in open(data_path):
            data = line.split("\t")
            if len(data) < 8 or data[6] is None or data[7] is None:
                continue
            yield (data[6], data[7])
