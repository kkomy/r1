import numpy as np

# -------------------------------------------------
# 0) 복제(동일값) 검사: (i,j,ratio) 목록 반환
# -------------------------------------------------
def detect_duplicates(messages, threshold=1.0):
    tokens = to_tokens_2d(messages)
    L = tokens.shape[1]
    dup = []
    for i in range(L):
        for j in range(i+1, L):
            ratio = float(np.mean(tokens[:, i] == tokens[:, j]))
            if ratio >= threshold:
                dup.append((i, j, ratio))
    return dup

# -------------------------------------------------
# 1) P(k): 예측가능성(파생성) = H(X_k | others) 근사
# -------------------------------------------------
def predictability_scores(messages, n_splits=5, seed=0):
    tokens = to_tokens_2d(messages)
    L = tokens.shape[1]
    cols = list(range(L))
    P = np.zeros(L)

    for k in range(L):
        feats = [c for c in cols if c != k]
        P[k] = cv_logloss(tokens, feats, k, n_splits, seed)

    return P

# -------------------------------------------------
# 2) Seq-U(k): 현재 전문 -> 다음 전문 예측에서의 기여도
#    U_seq(k) = Σ_j [ H(X_{t+1,j} | X_t \ {k}) - H(X_{t+1,j} | X_t) ]
# -------------------------------------------------
def seq_utility_scores(messages, n_splits=5, seed=0):
    tokens = to_tokens_2d(messages)
    N, L = tokens.shape
    if N < 2:
        return np.zeros(L)

    X_cur = tokens[:-1, :]
    X_next = tokens[1:, :]
    cols = list(range(L))

    # baseline: H(X_{t+1,j} | X_t)
    base_loss_j = np.array([
        _cv_logloss_on_arrays(X_cur, X_next[:, j], n_splits, seed)
        for j in range(L)
    ])

    SU = np.zeros(L)
    for k in cols:
        feats_wo = [c for c in cols if c != k]
        X_wo = X_cur[:, feats_wo]

        loss_j = np.array([
            _cv_logloss_on_arrays(X_wo, X_next[:, j], n_splits, seed)
            for j in range(L)
        ])

        SU[k] = float(np.sum(loss_j - base_loss_j))

    return SU

def _cv_logloss_on_arrays(X, y, n_splits=5, seed=0):
    """
    네 cv_logloss는 'tokens + cols' 형태라서,
    Seq-U에서는 (X,y) 배열 직접 넣는 전용 wrapper가 필요함.
    내부 로직은 네 cv_logloss와 동일(OneHot + LR + CV logloss)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    losses = []

    for tr, te in kf.split(X):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_tr_oh = enc.fit_transform(X_tr)
        X_te_oh = enc.transform(X_te)

        clf = LogisticRegression(max_iter=3000)
        clf.fit(X_tr_oh, y_tr)

        proba = clf.predict_proba(X_te_oh)
        losses.append(log_loss(y_te, proba, labels=clf.classes_))

    return float(np.mean(losses))

# -------------------------------------------------
# 3) 3-layer 파이프라인 (정답 없이) + 로그
#    - DUP STOP
#    - U<0면 확정
#    - U topK 후보 -> U+P로 정제 -> (필요시) Seq-U
# -------------------------------------------------
def find_dummy_index_3layer(
    messages,
    duplicate_threshold=1.0,  # 1.0: 완전복제만 STOP, 0.99: 거의복제도 STOP
    topK=3,                   # U에서 후보 몇 개 볼지
    up_keep=2,                # U+P에서 다음으로 넘길 후보 수(보통 1~2)
    eps_mode="auto",          # "auto" 또는 float (U< -eps면 확정)
    tie_delta=0.0,            # Seq-U에서 동률 판정 여유
    n_splits=5,
    seed=0
):
    tokens = to_tokens_2d(messages)
    L = tokens.shape[1]
    pos = "abcdefghijklmnopqrstuvwxyz"[:L]

    log = {
        "decided_at": None,    # DUPLICATE | NEG_U | UP | SEQ | SEQ_AMBIGUOUS
        "reason": None,
        "duplicates": None,
        "candidates": {},
        "scores": {}
    }

    # 0) 복제 검사
    dup = detect_duplicates(messages, threshold=duplicate_threshold)
    if dup:
        log["decided_at"] = "DUPLICATE"
        log["reason"] = "자리쌍이 (완전/거의) 동일값 → 전문만으로 식별 불가 가능성 큼"
        log["duplicates"] = [(pos[i], pos[j], r) for i, j, r in dup]
        return None, log  # STOP

    # 1) U (네 기존 함수 재사용)
    dummy_u, U = find_dummy_index(messages, n_splits=n_splits, seed=seed)
    log["scores"]["U"] = {pos[i]: float(U[i]) for i in range(L)}

    # eps 설정 (너 관찰: U 음수면 강력. 아주 작은 음수는 잡음일 수 있어 eps 둠)
    eps = max(1e-3, 0.1 * float(np.std(U))) if eps_mode == "auto" else float(eps_mode)
    log["eps"] = eps

    if float(np.min(U)) < -eps:
        idx = int(np.argmin(U))
        log["decided_at"] = "NEG_U"
        log["reason"] = f"min(U)={float(np.min(U)):.6f} < -eps={-eps:.6f} → U 음수 확정"
        return idx, log

    # 1-2) U 후보 topK
    K = min(max(1, topK), L)
    candU = list(np.argsort(U)[:K])
    log["candidates"]["U_topK"] = [pos[i] for i in candU]
    log["reason"] = f"U 음수 확정 없음 → U 상위 {K}개 후보로 진행"

    # 2) U+P (후보 집합에서 rank-sum)
    P = predictability_scores(messages, n_splits=n_splits, seed=seed)
    log["scores"]["P"] = {pos[i]: float(P[i]) for i in range(L)}

    Usub = np.array([U[i] for i in candU])
    Psub = np.array([P[i] for i in candU])
    score_up = Usub.argsort().argsort() + Psub.argsort().argsort()

    keep = min(max(1, up_keep), len(candU))
    order = np.argsort(score_up)[:keep]
    cand2 = [candU[int(o)] for o in order]
    log["candidates"]["UP_keep"] = [pos[i] for i in cand2]
    log["scores"]["UP_rank_sum_on_candidates"] = {
        pos[candU[i]]: int(score_up[i]) for i in range(len(candU))
    }

    # 후보 1개면 U+P에서 확정
    if len(cand2) == 1:
        idx = cand2[0]
        log["decided_at"] = "UP"
        log["reason"] = "U+P에서 후보 1개로 정리 → 확정"
        return idx, log

    # 3) Seq-U로 최종 분리
    SU = seq_utility_scores(messages, n_splits=n_splits, seed=seed)
    log["scores"]["Seq_U"] = {pos[i]: float(SU[i]) for i in range(L)}
    log["candidates"]["SEQ_compared"] = [pos[i] for i in cand2]
    log["candidates"]["SEQ_values"] = {pos[i]: float(SU[i]) for i in cand2}

    vals = np.array([SU[i] for i in cand2])
    best = cand2[int(np.argmin(vals))]

    sv = np.sort(vals)
    if len(sv) >= 2 and (sv[1] - sv[0] <= tie_delta):
        log["decided_at"] = "SEQ_AMBIGUOUS"
        log["reason"] = "Seq-U 후보 간 차이가 너무 작음 → 전문만으로 애매"
        return best, log

    log["decided_at"] = "SEQ"
    log["reason"] = "Seq-U로 후보 최종 분리 → 확정"
    return best, log

idx, info = find_dummy_index_3layer(messages, duplicate_threshold=0.99, topK=3, up_keep=2)

print("dummy_idx:", idx)
print("decided_at:", info["decided_at"])
print("reason:", info["reason"])
print("U_topK:", info["candidates"].get("U_topK"))
print("UP_keep:", info["candidates"].get("UP_keep"))
