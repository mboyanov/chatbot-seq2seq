def parse(udc_entry):
    turns = udc_entry[0].strip().split('__eot__')
    turns += [udc_entry[1]]
    utterances = ["".join(turn.split("__eou__")) for turn in turns]
    return utterances
