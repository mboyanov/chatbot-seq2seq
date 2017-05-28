def calculateMAP(similarities, correct):
    results = list(zip(similarities, correct))
    results.sort(reverse=True, key=lambda x: x[0])
    relevant_docs = 0
    score = 0.0
    for i, r in enumerate(results):
        if r[1]:
            relevant_docs += 1
            score += relevant_docs / (i + 1)
    return score / relevant_docs