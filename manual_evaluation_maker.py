import tensorflow as tf
from chatbot import decode
import pandas as pd

questions = []
for line in open('textquestions-all.txt'):
    questions.append(line.strip())


def generateResponses(metric, model_version, iteration):
    with tf.Session() as sess, open('manual_eval/%s_manual_test.csv' % metric,'w') as out:
        model_path = "/mnt/8C24EDC524EDB1FE/data/model_dir_ver_ranlp%s/translate.ckpt-%s" % (model_version, iteration)
        _, responder_multiple = decode(sess,model_path=model_path)
        responses = responder_multiple(questions)
        for q, response in zip(questions, responses):
            out.write("\t".join([q, response]) + "\n")

ds = pd.read_csv('manual_evaluation_maker_requirements.csv')
for i,row in ds.iterrows():
    model = '%s-%s-31.05' % (row.metric, 'ver3')
    generateResponses(model, '3', row.ver3)
    tf.reset_default_graph()