import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve


def thermodynamics(logits, T=1.0, eps=1e-12):
    z = np.exp(logits / T)
    p = z / z.sum(axis=1, keepdims=True)
    U = np.sum(p * (-logits), axis=1)
    S = -np.sum(p * np.log(p + eps), axis=1)
    F = -T * np.log(z.sum(axis=1))
    return F, U, S


def evaluate_score(y, score, name):
    s = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)

    roc_auc = roc_auc_score(y, s)
    pr_auc = average_precision_score(y, s)

    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = np.argmax(f1)

    id_scores = s[y == 0]
    ood_scores = s[y == 1]

    return {
        'score': name,
        'values': s,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_f1': f1[best_idx],
        'best_threshold': thresholds[best_idx],
        'id_mean': id_scores.mean(),
        'ood_mean': ood_scores.mean(),
        'mean_gap_ood_minus_id': ood_scores.mean() - id_scores.mean(),
        'id_std': id_scores.std(),
        'ood_std': ood_scores.std(),
    }


def pairwise_masks(id_scores, ood_scores):
    return id_scores[:, None], ood_scores[None, :]


def pairwise_summary(id_scores, ood_scores, id_scores_ref=None, ood_scores_ref=None):
    a_id, a_ood = pairwise_masks(id_scores, ood_scores)
    correct = a_id < a_ood

    out = {
        'total_pairs': correct.size,
        'correct_pairs': correct.sum(),
        'correct_ratio': correct.mean(),
    }

    if id_scores_ref is not None and ood_scores_ref is not None:
        b_id, b_ood = pairwise_masks(id_scores_ref, ood_scores_ref)
        ref_wrong = b_id > b_ood
        both = correct & ref_wrong
        out.update({
            'correct_and_ref_wrong': both.sum(),
            'correct_and_ref_wrong_ratio': both.mean(),
            'correct_and_ref_wrong_among_correct': both.sum() / (correct.sum() + 1e-12),
            'correct_and_ref_wrong_among_ref_wrong': both.sum() / (ref_wrong.sum() + 1e-12),
        })

    return out


def top_discordant_pairs(
    logits_id, logits_ood,
    score_good_id, score_good_ood,
    score_bad_id, score_bad_ood,
    top_k=5
):
    good_correct = score_good_id[:, None] < score_good_ood[None, :]
    bad_wrong = score_bad_id[:, None] > score_bad_ood[None, :]
    mask = good_correct & bad_wrong

    i_idx, j_idx = np.where(mask)
    if len(i_idx) == 0:
        return None

    bad_gap = score_bad_id[:, None] - score_bad_ood[None, :]
    good_gap = score_good_ood[None, :] - score_good_id[:, None]

    pairs = pd.DataFrame({
        'id_idx_local': i_idx,
        'ood_idx_local': j_idx,
        'bad_gap_wrong': bad_gap[i_idx, j_idx],
        'good_gap_correct': good_gap[i_idx, j_idx],
        'good_id': score_good_id[i_idx],
        'good_ood': score_good_ood[j_idx],
        'bad_id': score_bad_id[i_idx],
        'bad_ood': score_bad_ood[j_idx],
    }).sort_values(['bad_gap_wrong', 'good_gap_correct'], ascending=[False, False]).reset_index(drop=True)

    print("\n=== Топ пар: canonical прав, alternative неправ ===")
    print(pairs.head(10).round(4))

    for rank, row in pairs.head(top_k).iterrows():
        i, j = int(row.id_idx_local), int(row.ood_idx_local)
        print(f"\n--- Pair #{rank + 1} ---")
        print(
            f"good(ID)={row.good_id:.4f}, good(OOD)={row.good_ood:.4f}, "
            f"good_gap_correct={row.good_gap_correct:.4f}"
        )
        print(
            f"bad(ID)={row.bad_id:.4f}, bad(OOD)={row.bad_ood:.4f}, "
            f"bad_gap_wrong={row.bad_gap_wrong:.4f}"
        )
        print("ID logits:")
        print(np.round(logits_id[i], 4))
        print("OOD logits:")
        print(np.round(logits_ood[j], 4))

    return pairs


# ===== Data =====
data = pd.read_csv('/Volumes/HP/proj/mnist_data/results/logits.csv')

T = 1.0
y = data.values[:, 0].astype(int)     # 0 = ID, 1 = OOD
logits = data.values[:, 2:]

F, U, S = thermodynamics(logits, T)

scores = {
    'U-TS': U - T * S,
    'U+TS': U + T * S,
}

results = [evaluate_score(y, score, name) for name, score in scores.items()]
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'values'} for r in results])
results_df = results_df.sort_values('roc_auc', ascending=False)

print('\n=== Separation results ===')
print(results_df.round(4))


# ===== ROC =====
plt.figure(figsize=(7, 6))
for r in results:
    fpr, tpr, _ = roc_curve(y, r['values'])
    plt.plot(fpr, tpr, label=f"{r['score']} ROC-AUC={r['roc_auc']:.4f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curves for ID vs OOD separation')
plt.legend()
plt.grid(True)
plt.show()


# ===== Histograms =====
plt.figure(figsize=(12, 5))
for idx, r in enumerate(results, start=1):
    plt.subplot(1, len(results), idx)
    s = r['values']
    plt.hist(s[y == 0], bins=50, alpha=0.6, label='ID')
    plt.hist(s[y == 1], bins=50, alpha=0.6, label='OOD')
    plt.xlabel(r['score'])
    plt.ylabel('Count')
    plt.title(f'Score distribution: {r["score"]}')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()


# ===== Pairwise =====
id_mask = y == 0
ood_mask = y == 1

logits_id, logits_ood = logits[id_mask], logits[ood_mask]
F_id, F_ood = F[id_mask], F[ood_mask]
P_id, P_ood = scores['U+TS'][id_mask], scores['U+TS'][ood_mask]

pair_stats = pairwise_summary(P_id, P_ood, F_id, F_ood)

print("\n=== Pairwise summary: U+TS correct, F wrong ===")
for k, v in pair_stats.items():
    print(f"{k}: {v}")


# ===== Top discordant pairs =====
top_discordant_pairs(
    logits_id=logits_id,
    logits_ood=logits_ood,
    score_good_id=F_id,
    score_good_ood=F_ood,
    score_bad_id=P_id,
    score_bad_ood=P_ood,
    top_k=5
)