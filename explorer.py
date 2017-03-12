import numpy as np


def explore(datasets, tokenizer, flat=False):
    for ds in datasets:
        stats = {}
        stats['ds'] = ds['name']
        stats['total_subword_units_q'] = 0
        stats['total_subword_units_a'] = 0

        length_q = []
        length_a = []
        for (q, answers) in ds['reader']:
            if flat:
                answers = [(answers, None)]
            subword_units_q = tokenizer(q)
            stats['total_subword_units_q'] += len(subword_units_q)
            length_q.append(len(subword_units_q))
            for a in answers:
                subword_units_a = tokenizer(a[0])
                stats['total_subword_units_a'] += len(subword_units_a)
                length_a.append(len(subword_units_a))
        stats['total_questions'] = len(length_q)
        stats['total_answers'] = len(length_a)
        stats['mean_q_length'] = np.mean(length_q)
        stats['std_q_length'] = np.std(length_q)
        stats['mean_a_length'] = np.mean(length_a)
        stats['std_a_length'] = np.std(length_a)
        stats['histogram_q_length'] = histogram(length_q)
        stats['histogram_a_length'] = histogram(length_a)
        ordered = ['ds',
                   'total_questions',
                   'total_answers',
                   "total_subword_units_q",
                   'total_subword_units_a',
                   'mean_q_length',
                   'std_q_length',
                   'mean_a_length',
                   'std_a_length',
                   'histogram_q_length',
                   'histogram_a_length']
        print('\t'.join(ordered))
        print('\t'.join(str(stats[o]) for o in ordered))


def histogram(length_q):

    counts, bins = np.histogram(length_q, bins=[0,5,10,15,20,30,50,100,200,500])
    bins = [str(b) for b in bins]
    bins = ["-".join(bins[i:i+2]) for i in range(len(bins)-1)]
    return list(zip(bins, counts))