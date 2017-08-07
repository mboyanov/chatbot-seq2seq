# Building Chatbots from Forum Data: Model Selection Using Question Answering Metrics

This project explores how we can leverage existing forum data to create a question answering chatbot which provides informative
answers. 

The description and result paper is available here: TODO.

Links to the training data are available in the following table:

| Dataset           | Description   | Number of (Q,A) pairs
| -------------     | ------------- | -------------
| [QatarLiving Full](https://drive.google.com/open?id=0BxYLkQRqdXrcVHI4UmJEYWFMUEE)  | Contains pairs of (Q,A) sentences derived from the QL dump via averaging embeddings| 1716698
| [QatarLiving Relevant](https://drive.google.com/file/d/0BxYLkQRqdXrcelctZUJtT2hhdGc/view?usp=sharing)  | Same as QatarLiving Full, but only contains the (Q,A) pairs where the answer A is deemed relevant to the question Q by a state of the art classifier| 525712

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
Coming soon
## Manual Evaluation

The manual evaluation was based on the following [100 questions and answers given by three of the models](https://drive.google.com/open?id=1tuaAPOlqyOC5qeyLinx52HKIN9ohZqre1j0bxh2ERl8). The first 50 questions are taken from the test dataset,
whereas the latter are new, but still related to the forum topics. The answers produced by the three systems for these 100 questions were evaluated independently by four annotators, who judged whether each of the answers is good.

| Optimizing for | Questions 1-50 | Questions 50-100 | Question 1-100 |
| -------------- | ---------------| -----------------| -------------- |
| MAP            |  49.50 %       |  45.00 %         | 47.25 %         |
| BLEU           | 21.00 %        | 11.00 %          | 16.00 %
| Seq2Seq Loss   | 46.50 %        | 37.00 %          | 41.75 %        | 
