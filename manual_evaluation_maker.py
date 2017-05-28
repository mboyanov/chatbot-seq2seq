import tensorflow as tf
from chatbot import decode

questions = []
for line in open('manual_test_questions.txt'):
    questions.append(line.strip())


model = 'BLEU_smaller'
with tf.Session() as sess, open('%s_manual_test.csv' % model,'w') as out:
    responder, responder_multiple = decode(sess)
    responses = responder_multiple(questions)
    for q, response in zip(questions, responses):
        out.write("\t".join([q, response]) + "\n")