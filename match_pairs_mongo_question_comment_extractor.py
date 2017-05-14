from db import db

def get_qc_pair(qc_id):
    res = db.matched_pairs.find({"comment_id": qc_id})
    if res.count() == 0:
        return None
    best = max(res, key=lambda x: x['relevance'])
    return best