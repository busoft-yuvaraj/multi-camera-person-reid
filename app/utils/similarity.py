import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.
    Assumes vectors might not be normalized (though OSNet vectors should be).
    """
    if a.shape != b.shape:
        return 0.0
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(np.dot(a, b) / (norm_a * norm_b))
