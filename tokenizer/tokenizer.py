import re
from collections import defaultdict
cleanr = re.compile('<.*?>')
_DIGIT_RE = re.compile(r"\d")



class Tokenizer:


    def __init__(self, unknown_symbol, vocab_list=[], normalize_digits = True, clean_apostrophes=True):
        self.normalize_digits = normalize_digits
        self.unknown_symbol = str(unknown_symbol)
        self.vocab = dict([(x, i) for (i, x) in enumerate(vocab_list)])
        self.rev_vocab = vocab_list
        self.unknown_symbol_id = -1
        self.clean_apostrophes = clean_apostrophes
        if self.unknown_symbol in self.vocab:
            self.unknown_symbol_id = self.vocab[self.unknown_symbol]

    def cleanhtml(self, raw_html):
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext

    def fit(self, dataset_iterator, max_tokens, reserved_list):
        vocab = defaultdict(lambda:0)
        for (q, a) in dataset_iterator:
            for t in self.tokenize(q):
                vocab[t] += 1
            for t in self.tokenize(a):
                vocab[t] += 1

        print("Total words %i" % len(vocab))
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        vocab_list = [(x, 999999999) for x in reserved_list] + sorted_vocab
        print("Top 20 ", vocab_list[:20])
        if len(vocab_list) > max_tokens:
            vocab_list = vocab_list[:max_tokens]
        return vocab_list



    def transform(self, sentence):
        tokens = self.tokenize(sentence)
        return [self._find_id(t) for t in tokens]

    def inverse_transform(self, token_ids):
        return [self.rev_vocab[i] for i in token_ids if i != 0]

    def _find_id(self, token):
        return self.vocab.get(token, self.unknown_symbol_id)

    def tokenize(self, sentence):
        words = []
        if self.normalize_digits:
            sentence = _DIGIT_RE.sub("0", sentence)
        if self.clean_apostrophes:
            sentence = re.sub(r'[iI]\'m', 'i am', sentence)
            sentence = re.sub(r'won\'t', 'will not', sentence)
        sentence = self.cleanhtml(sentence)
        for space_separated_fragment in sentence.strip().lower().split():
            if type(space_separated_fragment) == str:
                words.extend(re.split("([.,!?\"':;)(/\-]+)|[_*=]+", space_separated_fragment))
            else:
                words.extend(re.split("([.,!?\"':;)(]+)", space_separated_fragment))
        return [w for w in words if w]

class BPETokenizer(Tokenizer):

    def __init__(self, codes, start_vocab):
        Tokenizer.__init__(self, None)
        self.bpe_codes = [tuple(item.split()) for item in codes]
        self.bpe_codes = start_vocab + self.bpe_codes[:-4]
        self.codes = self.bpe_codes
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])
        self.singles = {}
        self.start_vocab = start_vocab

    def __call__(self, *args, **kwargs):
        return self.transform(*args)


    def transform(self, sentence):
        tokens = Tokenizer.tokenize(self, sentence)
        output = []
        for word in tokens:
            new_word, new_word_indices = self.encode(word, self.bpe_codes)
            while -1 in new_word_indices:
                index = new_word_indices.index(-1)
                output.extend(new_word_indices[:index])
                if new_word[index] == '</w>':
                    output.append(len(self.codes))
                elif ord(new_word[index]) < 256:
                    output.append(ord(new_word[index]) + len(self.codes) + 1)
                else:
                    print("skipping character", new_word[index])
                new_word_indices = new_word_indices[index+1:]
                new_word = new_word[index+1:]
            output.extend(new_word_indices)
        return output

    def inverse_transform(self, token_ids):
        output = []
        for token in token_ids:
            if token < len(self.start_vocab):
                output.append(str(self.start_vocab[token]))
            elif token < len(self.codes):
                bp = self.codes[token]
                output.append("".join(b.replace('</w>', ' ') for b in bp))
            elif token == len(self.codes):
                output.append(' ')
            else:
                output.append(chr(token - len(self.codes) - 1))
        return output


    def get_pairs(self, word):
        """Return set of symbol pairs in a word.

        word is represented as tuple of symbols (symbols being variable-length strings)
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def encode(self, orig, bpe_codes):
        """Encode word based on list of BPE merge operations, which are applied consecutively
        """

        word = tuple(orig) + ('</w>',)
        word_indices = [-1] * (len(word))
        pairs = self.get_pairs(word)

        while True:
            bigram = min(pairs, key=lambda pair: bpe_codes.get(pair, float('inf')))
            if bigram not in bpe_codes:
                break
            first, second = bigram
            new_word = []
            new_word_indices = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    new_word_indices.extend(word_indices[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    new_word_indices.extend(word_indices[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first+second)
                    new_word_indices.append(bpe_codes[bigram])
                    i += 2
                else:
                    new_word.append(word[i])
                    new_word_indices.append(word_indices[i])
                    i += 1
            new_word = tuple(new_word)
            new_word_indices = tuple(new_word_indices)
            word = new_word
            word_indices = new_word_indices
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        return word, word_indices


a = BPETokenizer(open("/home/martin/projects/subword-nmt/vocab_bpe"), [b"GO", b"UNK", b"EOS", b"PAD"])
t = a.transform("i'm considering to take on a job opportunity in doha however as i am married with my male partner who is not a european i would need to know if i could have him granted a visa on the gounds of him being my legal partner and if my life would turn into a living hell once in qatar .")
#t = a.transform("electroworltdltd00@yahoo gounds")
t2 = a.inverse_transform(t+[1])
inp = [122,1172,63,33,2866,234,158,2177,73,4,3,2,1,0]
t3 = a.inverse_transform(inp)
print(t3)
print (t)
print(t2)