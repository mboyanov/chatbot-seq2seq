import re
import QLXMLReaderPy
import numpy as np
def explore(datasets):
    for ds in datasets:
        length_q = []
        length_a = []
        for (q, answers) in QLXMLReaderPy.read(ds):
            length_q.append(len(re.split("\s", q)))
            for a in answers:
                length_a.append(len(re.split("\s", a[0])))
        print(ds, "stats-q", len(length_q), np.mean(length_q), np.std(length_q))
        print(ds, "stats-a", len(length_a), np.mean(length_a), np.std(length_a))