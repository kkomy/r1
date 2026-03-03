import numpy as np
from collections import defaultdict

def build_matrix(data, split):
    """
    data: list of 4-digit strings
    split: int (1,2,3) meaning L size
    """
    L_size = split
    R_size = 4 - split
    
    pairs = []
    L_set = set()
    R_set = set()
    
    for s in data:
        L = s[:L_size]
        R = s[L_size:]
        pairs.append((L, R))
        L_set.add(L)
        R_set.add(R)
    
    L_list = sorted(list(L_set))
    R_list = sorted(list(R_set))
    
    L_index = {v:i for i,v in enumerate(L_list)}
    R_index = {v:i for i,v in enumerate(R_list)}
    
    A = np.zeros((len(L_list), len(R_list)))
    
    for L, R in pairs:
        A[L_index[L], R_index[R]] = 1
        
    return A

def rank_score(A):
    """
    return rank-1 explained ratio
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    total_energy = np.sum(S**2)
    explained = S[0]**2
    return explained / total_energy

def evaluate_splits(data):
    splits = [1,2,3]
    results = {}
    
    for s in splits:
        A = build_matrix(data, s)
        score = rank_score(A)
        results[f"{s}|{4-s}"] = score
        
    return results
