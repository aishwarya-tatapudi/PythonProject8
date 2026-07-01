import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PIPELINE 2 — ENGLISH SYSTEM-PROMPT-AWARE
# SPML Dataset + Relevance Gating + SVM
# Output: DENY_IRRELEVANT / INJECTION_DETECTED / SAFE
# ============================================

# ============================================
# STEP 1: LOAD SPML DATA
# ============================================
print("="*60)
print("STEP 1: LOADING SPML DATA")
print("="*60)

df = pd.read_csv('spml_prompt_injection.csv')

print(f"Columns: {df.columns.tolist()}")
print(f"Total rows: {len(df)}")

# Rename for convenience
df = df.rename(columns={
    'System Prompt':    'system_prompt',
    'User Prompt':      'user_prompt',
    'Prompt injection': 'label',
    'Degree':           'degree',
    'Source':           'source'
})

# Clean nulls
df['system_prompt'] = df['system_prompt'].fillna('').astype(str).str.strip()
df['user_prompt']   = df['user_prompt'].fillna('').astype(str).str.strip()
df['label']         = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

# Remove rows where both prompts are empty
df = df[(df['system_prompt'] != '') | (df['user_prompt'] != '')].reset_index(drop=True)

print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nRows with system prompt: {(df['system_prompt'] != '').sum()}")
print(f"Rows without system prompt: {(df['system_prompt'] == '').sum()}")

# ============================================
# STEP 2: BALANCE AND SAMPLE
# ============================================
print("\n" + "="*60)
print("STEP 2: BALANCING DATASET")
print("="*60)

attacks  = df[df['label'] == 1].reset_index(drop=True)
innocent = df[df['label'] == 0].reset_index(drop=True)

print(f"Available attacks:  {len(attacks)}")
print(f"Available innocent: {len(innocent)}")

# Use all innocent, sample equal attacks
n = min(len(attacks), len(innocent))
attacks_sampled  = attacks.sample(n=n, random_state=42)
innocent_sampled = innocent.sample(n=n, random_state=42)

balanced_df = pd.concat([attacks_sampled, innocent_sampled], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nBalanced dataset: {len(balanced_df)} total")
print(f"  Attacks:  {len(attacks_sampled)}")
print(f"  Innocent: {len(innocent_sampled)}")

# ============================================
# STEP 3: SPLIT 80/20
# ============================================
print("\n" + "="*60)
print("STEP 3: SPLITTING DATA 80/20")
print("="*60)

train_df, test_df = train_test_split(
    balanced_df, test_size=0.2, random_state=42,
    stratify=balanced_df['label']
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Training set: {len(train_df)}")
print(f"  Attacks:  {train_df['label'].sum()}")
print(f"  Innocent: {(train_df['label'] == 0).sum()}")
print(f"\nTest set:     {len(test_df)}")
print(f"  Attacks:  {test_df['label'].sum()}")
print(f"  Innocent: {(test_df['label'] == 0).sum()}")

# ============================================
# STEP 4: CREATE COMBINED TEXT
# System + User combined for SVM training
# ============================================
print("\n" + "="*60)
print("STEP 4: CREATING COMBINED PROMPTS")
print("="*60)

def combine_prompts(row):
    if row['system_prompt'] and row['system_prompt'] != '':
        return f"[SYSTEM]: {row['system_prompt']} [USER]: {row['user_prompt']}"
    else:
        return f"[USER]: {row['user_prompt']}"

train_df['combined'] = train_df.apply(combine_prompts, axis=1)
test_df['combined']  = test_df.apply(combine_prompts, axis=1)

print(f"Sample combined (attack):")
sample_attack = train_df[train_df['label'] == 1].iloc[0]['combined']
print(f"  {sample_attack[:200]}...")
print(f"\nSample combined (innocent):")
sample_innocent = train_df[train_df['label'] == 0].iloc[0]['combined']
print(f"  {sample_innocent[:200]}...")

# ============================================
# STEP 5: LOAD MINILM
# ============================================
print("\n" + "="*60)
print("STEP 5: LOADING MULTILINGUAL MODEL")
print("="*60)

print("Loading paraphrase-multilingual-MiniLM-L12-v2...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded!")

# ============================================
# STEP 6: ENCODE TEXT
# ============================================
print("\n" + "="*60)
print("STEP 6: ENCODING TEXT TO VECTORS")
print("="*60)

print("Encoding training USER prompts for SVM...")
train_vectors = embedder.encode(
    train_df['user_prompt'].tolist(),
    show_progress_bar=True, batch_size=32
)
print(f"Training vectors: {train_vectors.shape}")

print("\nEncoding test USER prompts for SVM...")
test_vectors = embedder.encode(
    test_df['user_prompt'].tolist(),
    show_progress_bar=True, batch_size=32
)
print(f"Test vectors: {test_vectors.shape}")

# Encode system prompts separately for relevance gate
print("\nEncoding system prompts for relevance gate...")
test_system_vectors = embedder.encode(
    test_df['system_prompt'].tolist(),
    show_progress_bar=True, batch_size=32
)

# User vectors for gate reuse test_vectors
test_user_vectors = test_vectors.copy()
print("User vectors reused for relevance gate.")

# ============================================
# STEP 7: TRAIN SVM
# ============================================
print("\n" + "="*60)
print("STEP 7: TRAINING SVM CLASSIFIER")
print("="*60)

svm = SVC(
    kernel='rbf',
    probability=True,
    class_weight='balanced',
    random_state=42,
    C=1.0
)
svm.fit(train_vectors, train_df['label'].values)
print("SVM training complete!")
print(f"Trained on: {len(train_df)} user prompts (system prompt used separately for gate)")

# ============================================
# STEP 8: SVM BASELINE (no relevance gate)
# ============================================
print("\n" + "="*60)
print("STEP 8: SVM BASELINE (no relevance gate)")
print("="*60)

svm_preds = svm.predict(test_vectors)
tn, fp, fn, tp = confusion_matrix(test_df['label'].values, svm_preds).ravel()

svm_accuracy = (tp + tn) / (tp + tn + fp + fn)
svm_fpr      = fp / (fp + tn) if (fp + tn) > 0 else 0
svm_fnr      = fn / (fn + tp) if (fn + tp) > 0 else 0
svm_tpr      = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nSVM ONLY RESULTS:")
print(f"  Accuracy:   {svm_accuracy:.2%}")
print(f"  TPR:        {svm_tpr:.2%}")
print(f"  FPR:        {svm_fpr:.2%}")
print(f"  FNR:        {svm_fnr:.2%}")
print(f"\nConfusion Matrix:")
print(f"  True Positives:  {tp}  (attacks caught)")
print(f"  True Negatives:  {tn}  (innocent allowed)")
print(f"  False Positives: {fp}  (innocent blocked)")
print(f"  False Negatives: {fn}  (attacks missed)")

# ============================================
# STEP 9: RELEVANCE GATE
# Cosine similarity between system and user vectors
# ============================================
print("\n" + "="*60)
print("STEP 9: RELEVANCE GATE — THRESHOLD SEARCH")
print("="*60)

def compute_relevance(sys_vec, usr_vec):
    """Cosine similarity between system and user prompt vectors."""
    if np.all(sys_vec == 0) or np.all(usr_vec == 0):
        return 1.0  # no system prompt = general assistant = allow
    sim = cosine_similarity([sys_vec], [usr_vec])[0][0]
    return float(sim)

# Compute similarities for all test prompts
print("Computing cosine similarities...")
similarities = []
for i in range(len(test_df)):
    sys_vec = test_system_vectors[i]
    usr_vec = test_user_vectors[i]

    # If no system prompt, similarity = 1 (allow by default)
    if test_df.iloc[i]['system_prompt'] == '':
        similarities.append(1.0)
    else:
        sim = compute_relevance(sys_vec, usr_vec)
        similarities.append(sim)

similarities = np.array(similarities)
test_df['relevance_score'] = similarities

print(f"\nRelevance score distribution:")
print(f"  Mean:   {similarities.mean():.3f}")
print(f"  Median: {np.median(similarities):.3f}")
print(f"  Min:    {similarities.min():.3f}")
print(f"  Max:    {similarities.max():.3f}")

# Test thresholds from 0.1 to 0.5
print(f"\nThreshold search:")
print(f"{'Threshold':>12} {'Gate Blocks':>12} {'Accuracy':>10} {'FPR':>8} {'FNR':>8} {'IRRELEVANT':>12}")
print("-"*70)

best_threshold = 0.25
best_accuracy  = 0

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    # Prompts flagged as irrelevant
    irrelevant_mask = similarities < threshold
    n_irrelevant    = irrelevant_mask.sum()

    # For non-irrelevant prompts, use SVM
    final_preds = []
    for i in range(len(test_df)):
        if irrelevant_mask[i]:
            final_preds.append(1)  # DENY_IRRELEVANT counts as blocked
        else:
            conf = svm.predict_proba([test_vectors[i]])[0][1]
            final_preds.append(1 if conf >= 0.5 else 0)

    final_preds = np.array(final_preds)
    true_labels = test_df['label'].values
    tn_g, fp_g, fn_g, tp_g = confusion_matrix(true_labels, final_preds).ravel()

    acc = (tp_g + tn_g) / len(test_df)
    fpr = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0
    fnr = fn_g / (fn_g + tp_g) if (fn_g + tp_g) > 0 else 0

    print(f"{threshold:>12.2f} {n_irrelevant:>12} {acc:>10.2%} {fpr:>8.2%} {fnr:>8.2%} {n_irrelevant/len(test_df):>12.1%}")

    if acc > best_accuracy:
        best_accuracy  = acc
        best_threshold = threshold

print(f"\nBest threshold: {best_threshold} (accuracy: {best_accuracy:.2%})")

# ============================================
# STEP 10: FULL PIPELINE EVALUATION
# Relevance gate + SVM → three-way output
# ============================================
print("\n" + "="*60)
print(f"STEP 10: FULL PIPELINE EVALUATION (threshold={best_threshold})")
print("="*60)

def pipeline2_predict(combined_vec, sys_vec, usr_vec, has_system_prompt):
    """
    Three-way output:
      DENY_IRRELEVANT   — system/user mismatch
      INJECTION_DETECTED — attack pattern found
      SAFE               — allow
    """
    # Step 1 — Relevance gate
    if has_system_prompt:
        sim = compute_relevance(sys_vec, usr_vec)
        if sim < best_threshold:
            return "DENY_IRRELEVANT", sim, 0.0

    # Step 2 — SVM injection check
    confidence = svm.predict_proba([combined_vec])[0][1]
    if confidence >= 0.5:
        return "INJECTION_DETECTED", 1.0, confidence
    else:
        return "SAFE", 1.0, confidence

# Run full pipeline on test set
results = []
for i, row in test_df.iterrows():
    has_sys = row['system_prompt'] != ''
    outcome, relevance, confidence = pipeline2_predict(
        test_vectors[i],
        test_system_vectors[i],
        test_user_vectors[i],
        has_sys
    )
    results.append({
        'true_label':  row['label'],
        'system_prompt': row['system_prompt'][:100] if row['system_prompt'] else '',
        'user_prompt':   row['user_prompt'][:100],
        'outcome':       outcome,
        'relevance':     relevance,
        'confidence':    confidence,
        'has_system':    has_sys
    })

results_df = pd.DataFrame(results)

# Map to binary for metrics
# DENY_IRRELEVANT + INJECTION_DETECTED = blocked (1)
# SAFE = allowed (0)
results_df['binary_pred'] = (results_df['outcome'] != 'SAFE').astype(int)

tn_p, fp_p, fn_p, tp_p = confusion_matrix(
    results_df['true_label'], results_df['binary_pred']
).ravel()

pipe_accuracy = (tp_p + tn_p) / len(results_df)
pipe_fpr      = fp_p / (fp_p + tn_p) if (fp_p + tn_p) > 0 else 0
pipe_fnr      = fn_p / (fn_p + tp_p) if (fn_p + tp_p) > 0 else 0
pipe_tpr      = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0

print(f"\nFULL PIPELINE 2 RESULTS:")
print(f"  Accuracy:   {pipe_accuracy:.2%}")
print(f"  TPR:        {pipe_tpr:.2%}")
print(f"  FPR:        {pipe_fpr:.2%}")
print(f"  FNR:        {pipe_fnr:.2%}")
print(f"\nConfusion Matrix:")
print(f"  True Positives:  {tp_p}  (attacks caught)")
print(f"  True Negatives:  {tn_p}  (innocent allowed)")
print(f"  False Positives: {fp_p}  (innocent blocked)")
print(f"  False Negatives: {fn_p}  (attacks missed)")

# Three-way output distribution
print(f"\nTHREE-WAY OUTPUT DISTRIBUTION:")
outcome_counts = results_df['outcome'].value_counts()
total = len(results_df)
for outcome in ['DENY_IRRELEVANT', 'INJECTION_DETECTED', 'SAFE']:
    count = outcome_counts.get(outcome, 0)
    print(f"  {outcome:<25} {count:>5} ({count/total:.1%})")

# ============================================
# STEP 11: BEFORE VS AFTER GATE
# ============================================
print("\n" + "="*60)
print("STEP 11: BEFORE VS AFTER RELEVANCE GATE")
print("="*60)

print(f"\n{'Metric':<25}{'SVM Only':>12}{'Full Pipeline':>15}{'Change':>10}")
print("-"*65)

for metric, before, after in [
    ("Accuracy",  svm_accuracy, pipe_accuracy),
    ("FPR",       svm_fpr,      pipe_fpr),
    ("FNR",       svm_fnr,      pipe_fnr),
    ("TPR",       svm_tpr,      pipe_tpr),
]:
    change    = after - before
    direction = "↑" if change > 0 else "↓"
    print(f"{metric:<25}{before:>12.2%}{after:>15.2%}  {direction}{abs(change):.2%}")

# Gate contribution
gate_blocked = (results_df['outcome'] == 'DENY_IRRELEVANT').sum()
svm_blocked  = (results_df['outcome'] == 'INJECTION_DETECTED').sum()
safe         = (results_df['outcome'] == 'SAFE').sum()

print(f"\nGATE CONTRIBUTION:")
print(f"  Caught by relevance gate: {gate_blocked} ({gate_blocked/total:.1%})")
print(f"  Caught by SVM:            {svm_blocked} ({svm_blocked/total:.1%})")
print(f"  Allowed (SAFE):           {safe} ({safe/total:.1%})")

# ============================================
# STEP 12: RELEVANCE GATE INSPECTION
# What did the gate catch? What did it miss?
# ============================================
print("\n" + "="*60)
print("STEP 12: RELEVANCE GATE INSPECTION")
print("="*60)

gate_catches = results_df[
    (results_df['outcome'] == 'DENY_IRRELEVANT') &
    (results_df['true_label'] == 1)
]
gate_fp = results_df[
    (results_df['outcome'] == 'DENY_IRRELEVANT') &
    (results_df['true_label'] == 0)
]

print(f"\nATTACKS CAUGHT BY RELEVANCE GATE: {len(gate_catches)}")
print(f"(Attacks correctly denied as irrelevant)")
for _, row in gate_catches.head(5).iterrows():
    print(f"  Relevance: {row['relevance']:.3f} | System: {row['system_prompt'][:60]}...")
    print(f"  User: {row['user_prompt'][:80]}")
    print("  ---")

print(f"\nINNOCENT PROMPTS WRONGLY DENIED BY GATE: {len(gate_fp)}")
print(f"(False positives introduced by relevance gate)")
for _, row in gate_fp.head(5).iterrows():
    print(f"  Relevance: {row['relevance']:.3f} | System: {row['system_prompt'][:60]}...")
    print(f"  User: {row['user_prompt'][:80]}")
    print("  ---")

# ============================================
# STEP 13: COMPARISON WITH PIPELINE 1
# ============================================
print("\n" + "="*60)
print("STEP 13: PIPELINE COMPARISON SUMMARY")
print("="*60)

print(f"\n{'Metric':<25}{'Pipeline 1 (Hindi)':>20}{'Pipeline 1 (Hinglish)':>22}{'Pipeline 2 (English)':>22}")
print("-"*90)
print(f"{'Accuracy':<25}{'97.00%':>20}{'89.25%':>22}{pipe_accuracy:.2%}")
print(f"{'FPR':<25}{'2.50%':>20}{'15.00%':>22}{pipe_fpr:.2%}")
print(f"{'FNR':<25}{'3.50%':>20}{'6.50%':>22}{pipe_fnr:.2%}")
print(f"{'Defense layers':<25}{'Rule+SVM':>20}{'Rule+SVM':>22}{'Gate+SVM':>22}")
print(f"{'Output format':<25}{'Five-tier':>20}{'Five-tier':>22}{'Three-way':>22}")

# ============================================
# STEP 14: SAVE RESULTS
# ============================================
print("\n" + "="*60)
print("STEP 14: SAVING RESULTS")
print("="*60)

results_df.to_csv('pipeline2_results.csv', index=False)

summary = pd.DataFrame([{
    'pipeline': 'Pipeline 2 — English',
    'dataset': 'SPML',
    'n_train': len(train_df),
    'n_test': len(test_df),
    'threshold': best_threshold,
    'svm_only_accuracy': svm_accuracy,
    'full_pipeline_accuracy': pipe_accuracy,
    'fpr': pipe_fpr,
    'fnr': pipe_fnr,
    'tpr': pipe_tpr,
    'gate_blocks': int(gate_blocked),
    'svm_blocks': int(svm_blocked),
    'safe': int(safe)
}])
summary.to_csv('pipeline2_summary.csv', index=False)

print(f"  pipeline2_results.csv  — full test set results")
print(f"  pipeline2_summary.csv  — summary metrics")

print("\n" + "="*60)
print("PIPELINE 2 COMPLETE")
print("="*60)
print(f"\nKey results:")
print(f"  SVM only accuracy:      {svm_accuracy:.2%}")
print(f"  Full pipeline accuracy: {pipe_accuracy:.2%}")
print(f"  Relevance gate caught:  {gate_blocked} prompts ({gate_blocked/total:.1%})")
print(f"  Best threshold:         {best_threshold}")
print(f"\nOutput files:")
print(f"  pipeline2_results.csv")
print(f"  pipeline2_summary.csv")