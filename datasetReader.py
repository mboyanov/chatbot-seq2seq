class DatasetReader():
    def words(self, data_path, tokenizer):
        raise NotImplementedError()

    def conversations(self, data_path):
        raise NotImplementedError()
