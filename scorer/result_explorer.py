import pandas as pd
import sys
dev_file = sys.argv[1]
test_file = sys.argv[2]


def read_csv(file):
    return pd.read_csv(file, sep='\t')


dev_scores = read_csv(dev_file)
test_scores = read_csv(test_file)

test_scores.rename(columns=lambda x: x+"_TEST", inplace=True)

max_dev_scores = dev_scores.max()

with open('/tmp/combination', 'w') as out:
    for key in max_dev_scores.keys():
        max_for_this_key = dev_scores[dev_scores[key] == max_dev_scores[key]]
        iteration = max_for_this_key['Global step'].values[0]
        test_scores_for_this_key = test_scores[test_scores['Global step_TEST'] == iteration]
        combined = max_for_this_key.join(test_scores_for_this_key)
        out.write("\t".join([str(key), str(max_dev_scores[key]),str(test_scores_for_this_key[key+"_TEST"].values[0])]))
        out_csv = combined.to_csv(sep="\t", header=False,index=False)
        print(out_csv)
        out.write(out_csv[1:])