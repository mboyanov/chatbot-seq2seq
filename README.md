# Building Chatbots from Forum Data: Model Selection Using Question Answering Metrics

This project explores how we can leverage existing forum data to create a question answering chatbot which provides informative
answers. 

The description and result paper is available here: TODO.

Links to the training data are available in the following table:

| Dataset           | Description   | Number of (Q,A) pairs
| -------------     | ------------- | -------------
| [QatarLiving Full](https://drive.google.com/open?id=0BxYLkQRqdXrcVHI4UmJEYWFMUEE)  | Contains pairs of (Q,A) sentences derived from the QL dump via averaging embeddings| 1716698
| [QatarLiving Relevant](https://drive.google.com/file/d/0BxYLkQRqdXrcelctZUJtT2hhdGc/view?usp=sharing)  | Same as QatarLiving Full, but only contains the (Q,A) pairs where the answer A is deemed relevant to the question Q by a state of the art classifier| 525712


In order to evaluate the model you will also need the [evaluation data](http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-cqa-ql-traindev-v3.2.zip) from the Semeval Task. The datasets can be summarized as follows:

| Dataset           | Description   | Number of (Q,A) pairs
| -------------     | ------------- | -------------
| QatarLiving Dev   | Golden relevancy judgments used for model selection | 2440
| QatarLiving Test  | Golden relevancy judgments used for final evaluation| 3270


To run the model, one also need a BPE vocabulary. The one we used is available [here](https://drive.google.com/file/d/0BxYLkQRqdXrcWTNiaW5HVWYtcHc/view?usp=sharing).

# Description
The project wraps the TensorFlow seq2seq implementation to generate responses to questions posed by users in an online forum.
The procedure is as follows:
1. Data is cleaned and tokenized
2. The Byte Pair Encoding algorithm is applied to learn subwords
3. The resulting sequences are fed to a classical seq2seq model with attention
4. At every N timesteps the training procedure is paused for evaluation.

The main focus of the project is in the evaluation step. Evaluating chatbots based on BLEU tends to overfit on the language model.
Instead, we are interested in creating a conversational interface which will actually answer the question.

To select such a model we propose evaluating the chatbot with respect to it's ability to solve the [SemEval Task 3](http://alt.qcri.org/semeval2017/task3/). The assumption is that informative
responses generated from the chatbot should be similar to those marked relevant by the annotators and dissimilar to those marked Bad. Thus, similarity
with the chatbot generated responses should provide a good relevancy ranking. 

Thus, we propose using the ranking metric Mean Average Precision (MAP) to evaluate the chatbot as it trains. The ranking produced from several similarity functions is studied:
 - cosine - the cosine distance between the tfidf vectors of the generated response and the original response.
 - bm25 - the bm25 similarity between the generated response and the original response.
 - cosine-embeddings - the cosine distance between the embeddings vectors of the generated response and the original response.
 - bleu - the BLEU score between the generated response and the original response.
 - average - the average of the former. 
 
# Results

## Automated Evaluation

We experimented with different model selection criteria to measure which one would be best at solving the Community Question Answering Task. Choosing the model achieving the best ranking on the dev set, produces the best ranking on the test set, but also achieves higher BLEU scores.

| Optimizing for | MAP            | BLEU |
| -------------- | ---------------| -----------------|
| MAP            |  63.45         |  9.18            |
| BLEU           |  62.64         |   8.16           |
| Seq2Seq Loss   |  62.81         |   7.00           |

## Manual Evaluation

The manual evaluation was based on the following [100 questions and answers given by three of the models](https://drive.google.com/open?id=1tuaAPOlqyOC5qeyLinx52HKIN9ohZqre1j0bxh2ERl8). The first 50 questions are taken from the test dataset,
whereas the latter are new, but still related to the forum topics. The answers produced by the three systems for these 100 questions were evaluated independently by four annotators, who judged whether each of the answers is good.

| Optimizing for | Questions 1-50 | Questions 50-100 | Question 1-100 |
| -------------- | ---------------| -----------------| -------------- |
| MAP            |  49.50 %       |  45.00 %         | 47.25 %        |
| BLEU           | 21.00 %        | 11.00 %          | 16.00 %        |
| Seq2Seq Loss   | 46.50 %        | 37.00 %          | 41.75 %        | 



# Pretrained models

We provide the following pretrained models. They are checkpoints of the same model, but selected by a different criteria. The parameters of the models were as follows:

1. vocabulary size: 40,000 subword units available [here](https://drive.google.com/open?id=0BxYLkQRqdXrcWTNiaW5HVWYtcHc);
2. dimensionality of the embedding vectors: 256;
3. RNN cell: 2-layered GRU cell with 256 units; 
4. minibatch size: 80; 
5. learning rate: 0.5; 
6. buckets: [(5, 10), (10, 15), (20, 25), (40,45)].


| Model | Metric | Iteration | 
| ------------------| ---------------| -----------------|
| [MAP-model](https://drive.google.com/open?id=0BxYLkQRqdXrcOXBUQi0yUjZ6Y2s)          |  MAP      |  192000         | 
| [BLEU-model](https://drive.google.com/open?id=0BxYLkQRqdXrcWFRkLUF0NlBXV00)         | BLEU         |  16000        | 
| [Seq2Seq Loss-model](https://drive.google.com/open?id=0BxYLkQRqdXrcVVRqWnVOTE1Ta3M) | Perplexity        | 200000          | 



