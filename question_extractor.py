from data_utils_udc import default_tokenizer, bpe_tokenizer
question_words = set(['how', 'when', 'who', 'where', 'what'])
class QuestionExtractor:


    def __init__(self, max_len, tokenizer=bpe_tokenizer):
        self.max_len = max_len
        self.tokenizer = tokenizer


    def extract(self, text):
        byte_codes = self.tokenizer.transform(text)
        if len(byte_codes) < self.max_len:
            return text
        tokens = self.tokenizer.tokenize(text)
        for i, token in enumerate(tokens):
            if token in question_words:
                return " ".join(tokens[i:i+self.max_len])
        return text