def precision_at_k(recommended_ids, ground_truth_ids, k=10):
    recommended_k = recommended_ids[:k]
    relevant = set(recommended_k) & set(ground_truth_ids)
    return len(relevant) / k

def recall_at_k(recommended_ids, ground_truth_ids, k=10):
    recommended_k = recommended_ids[:k]
    relevant = set(recommended_k) & set(ground_truth_ids)
    return len(relevant) / len(ground_truth_ids) if ground_truth_ids else 0

def f1_at_k(recommended_ids, ground_truth_ids, k=10):
    p = precision_at_k(recommended_ids, ground_truth_ids, k)
    r = recall_at_k(recommended_ids, ground_truth_ids, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0
