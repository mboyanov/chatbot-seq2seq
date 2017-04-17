import data_utils_udc
import trainingFilesReader
class ExecutionPlan:

    def __init__(self, udc_path, ql_path, udc_steps, ql_steps, vocab_size, _buckets, more_steps = []):
        self.steps = [
            {
                "ds": "udc",
                "num_steps": udc_steps,
                "path": udc_path
            },
            {
                "ds": "ql",
                "num_steps": ql_steps,
                "path": ql_path
            }

        ] + more_steps
        self.udc_path = udc_path
        self.ql_path = ql_path
        self.udc_steps = udc_steps
        self.ql_steps = ql_steps
        self.vocab_size = vocab_size
        self.current_data = {}
        self.buckets = _buckets


    def getData(self, step):
        acccumulated_step = 0
        for current_step in self.steps:
            if acccumulated_step <= step < acccumulated_step + current_step['num_steps']:
                print("training with %s at step %i out of %i steps" %
                      (current_step['ds'], step - acccumulated_step, current_step['num_steps']))
                questions_train, answers_train, questions_dev, answers_dev, _, _, _, = data_utils_udc.prepare_data(
                    current_step['path'], self.vocab_size, current_step['ds'])
                if current_step['ds'] not in self.current_data:
                    self.current_data[current_step['ds']] = read_data(questions_dev, answers_dev, self.buckets), read_data(questions_train, answers_train, self.buckets)
                return self.current_data[current_step['ds']]


def read_data(source_path, target_path, _buckets, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    for source, target in trainingFilesReader.read(source_path, target_path, max_size):
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils_udc.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
                data_set[bucket_id].append([source_ids, target_ids])
                break
    return data_set