def parse(udc_entry):
	if (udc_entry[-2:] == ',0'):
		return None
	turns = udc_entry[:-2].split('__eot__')
	utterances = ["".join(turn.split("__eou__")) for turn in turns]
	return utterances
