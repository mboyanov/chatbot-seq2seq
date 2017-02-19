import re
def tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        if type(space_separated_fragment) == 'str':
            words.extend(re.split("([.,!?\"':;)(])", space_separated_fragment))
        else:
            words.extend(re.split("([.,!?\"':;)(])", space_separated_fragment))
    return [w for w in words if w]
