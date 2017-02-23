import re

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens and lowercase."""
    words = []
    sentence = cleanhtml(sentence)
    for space_separated_fragment in sentence.strip().lower().split():
        if type(space_separated_fragment) == str:
            words.extend(re.split("[.,!?\"':;)(/=\-_*]+", space_separated_fragment))
        else:
            words.extend(re.split("([.,!?\"':;)(]+)", space_separated_fragment))
    return [w for w in words if w]


