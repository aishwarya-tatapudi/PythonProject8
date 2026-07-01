import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD DATA
# ============================================
print("="*60)
print("STEP 1: LOADING DATA")
print("="*60)

try:
    df = pd.read_excel(
        'PromptInjectionPrompts.xlsx',
        engine='openpyxl'
    )
except FileNotFoundError:
    df = pd.read_excel(
        'PromptInjectionPrompts.xls',
        engine='xlrd'
    )

print(f"Columns found: {df.columns.tolist()}")

df = df.rename(columns={
    'Prompts': 'text',
    'Label': 'label',
    'Language': 'language'
})

required_cols = ['text', 'label', 'language']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns after rename: {missing}. "
        f"Check that your Excel headers match: 'Prompts', 'Label', 'Language'"
    )

# Keep Translation column if it exists
if 'Translation' in df.columns:
    df = df[['text', 'label', 'language', 'Translation']]
else:
    df = df[required_cols]

print(f"Total examples: {len(df)}")
print(f"\nLanguage distribution:")
print(df['language'].value_counts())
print(f"\nLabel distribution:")
print(df['label'].value_counts())

hindi_attacks = df[(df['language'] == 'Hindi') & (df['label'] == 1)]['text']
hindi_innocent = df[(df['language'] == 'Hindi') & (df['label'] == 0)]['text']

if not hindi_attacks.empty:
    print(f"\nSample Hindi attack:")
    print(hindi_attacks.iloc[0])
else:
    print("\nNo Hindi attack samples found.")

if not hindi_innocent.empty:
    print(f"\nSample Hindi innocent:")
    print(hindi_innocent.iloc[0])
else:
    print("\nNo Hindi innocent samples found.")

# ============================================
# STEP 2: SPLIT DATA
# ============================================
print("\n" + "="*60)
print("STEP 2: SPLITTING DATA 80/20")
print("="*60)

hindi_df    = df[df['language'] == 'Hindi'].copy()
hinglish_df = df[df['language'] == 'Hinglish'].copy()

print(f"Hindi total: {len(hindi_df)}")
print(f"Hinglish total: {len(hinglish_df)}")

hi_train, hi_test = train_test_split(
    hindi_df,
    test_size=0.2,
    random_state=42,
    stratify=hindi_df['label']
)

hg_train, hg_test = train_test_split(
    hinglish_df,
    test_size=0.2,
    random_state=42,
    stratify=hinglish_df['label']
)

train_df = pd.concat([hi_train, hg_train], ignore_index=True)

print(f"\nTraining set: {len(train_df)}")
print(f"  Hindi train: {len(hi_train)}")
print(f"  Hinglish train: {len(hg_train)}")
print(f"\nTest sets:")
print(f"  Hindi test: {len(hi_test)}")
print(f"  Hinglish test: {len(hg_test)}")

# ============================================
# STEP 3: LOAD SENTENCE TRANSFORMER
# ============================================
print("\n" + "="*60)
print("STEP 3: LOADING MULTILINGUAL MODEL")
print("="*60)

print("Loading paraphrase-multilingual-MiniLM-L12-v2...")
print("(This understands Hindi and Hinglish meaning)")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded!")

# ============================================
# STEP 4: CONVERT TEXT TO MEANING VECTORS
# ============================================
print("\n" + "="*60)
print("STEP 4: CONVERTING TEXT TO MEANING VECTORS")
print("="*60)

print("Converting training data...")
train_vectors = embedder.encode(
    train_df['text'].tolist(),
    show_progress_bar=True,
    batch_size=32
)
print(f"Training vectors shape: {train_vectors.shape}")

print("\nConverting Hindi test data...")
hi_test_vectors = embedder.encode(
    hi_test['text'].tolist(),
    show_progress_bar=True,
    batch_size=32
)

print("\nConverting Hinglish test data...")
hg_test_vectors = embedder.encode(
    hg_test['text'].tolist(),
    show_progress_bar=True,
    batch_size=32
)

# ============================================
# STEP 5: TRAIN SVM
# ============================================
print("\n" + "="*60)
print("STEP 5: TRAINING SVM CLASSIFIER")
print("="*60)

print("Training SVM on Hindi + Hinglish data...")
svm = SVC(
    kernel='rbf',
    probability=True,
    class_weight='balanced',
    random_state=42,
    C=1.0
)
svm.fit(train_vectors, train_df['label'].values)
print("SVM training complete!")

# ============================================
# STEP 6: EVALUATE RESULTS
# ============================================
print("\n" + "="*60)
print("STEP 6: EVALUATING RESULTS")
print("="*60)

def evaluate(name, vectors, true_labels, test_df=None):
    """
    Evaluate model performance for one language.
    Shows translations for Hindi false positives and false negatives.
    """
    predictions = svm.predict(vectors)

    tn, fp, fn, tp = confusion_matrix(
        true_labels,
        predictions
    ).ravel()

    accuracy       = (tp + tn) / (tp + tn + fp + fn)
    fpr            = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr            = fn / (fn + tp) if (fn + tp) > 0 else 0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n{'='*40}")
    print(f"RESULTS: {name}")
    print(f"{'='*40}")
    print(f"Accuracy:                {accuracy:.2%}")
    print(f"Detection Rate (TPR):    {detection_rate:.2%}")
    print(f"False Positive Rate:     {fpr:.2%}")
    print(f"  (innocent users wrongly blocked)")
    print(f"False Negative Rate:     {fnr:.2%}")
    print(f"  (attacks that slipped through)")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp} (attacks correctly caught)")
    print(f"  True Negatives:  {tn} (innocent correctly allowed)")
    print(f"  False Positives: {fp} (innocent wrongly blocked)")
    print(f"  False Negatives: {fn} (attacks missed)")

    # Show misclassified prompts with translations
    if test_df is not None:
        has_translation = 'Translation' in test_df.columns

        inspect = test_df.copy().reset_index(drop=True)
        inspect['prediction'] = predictions
        inspect['confidence'] = svm.predict_proba(vectors)[:, 1]

        # False Positives — innocent prompts wrongly blocked
        fp_df = inspect[
            (inspect['label'] == 0) &
            (inspect['prediction'] == 1)
        ].sort_values('confidence', ascending=False)

        if not fp_df.empty:
            print(f"\n  {'='*50}")
            print(f"  FALSE POSITIVES — {name}")
            print(f"  Innocent prompts wrongly blocked ({len(fp_df)} total)")
            print(f"  {'='*50}")
            for _, row in fp_df.iterrows():
                print(f"  Confidence: {row['confidence']:.2%}")
                print(f"  Prompt:     {row['text']}")
                if has_translation and pd.notna(row.get('Translation', None)):
                    print(f"  Translation:{row['Translation']}")
                print(f"  {'-'*40}")

        # False Negatives — attacks that slipped through
        fn_df = inspect[
            (inspect['label'] == 1) &
            (inspect['prediction'] == 0)
        ].sort_values('confidence', ascending=True)

        if not fn_df.empty:
            print(f"\n  {'='*50}")
            print(f"  FALSE NEGATIVES — {name}")
            print(f"  Attacks that slipped through ({len(fn_df)} total)")
            print(f"  {'='*50}")
            for _, row in fn_df.iterrows():
                print(f"  Confidence: {row['confidence']:.2%}")
                print(f"  Prompt:     {row['text']}")
                if has_translation and pd.notna(row.get('Translation', None)):
                    print(f"  Translation:{row['Translation']}")
                print(f"  {'-'*40}")

    return {
        'language':       name,
        'accuracy':       accuracy,
        'detection_rate': detection_rate,
        'fpr':            fpr,
        'fnr':            fnr,
        'tp':             int(tp),
        'tn':             int(tn),
        'fp':             int(fp),
        'fn':             int(fn),
        'n_samples':      len(true_labels)
    }

# Evaluate both languages — pass test_df for translation support
hindi_results    = evaluate("HINDI",    hi_test_vectors, hi_test['label'].values, hi_test)
hinglish_results = evaluate("HINGLISH", hg_test_vectors, hg_test['label'].values, hg_test)

# ============================================
# STEP 7: COMPARISON SUMMARY
# ============================================
print("\n" + "="*60)
print("STEP 7: FINAL COMPARISON")
print("="*60)

print(f"\n{'Metric':<30}{'Hindi':>12}{'Hinglish':>12}")
print("-"*55)
print(f"{'Accuracy':<30}{hindi_results['accuracy']:>12.2%}{hinglish_results['accuracy']:>12.2%}")
print(f"{'Detection Rate':<30}{hindi_results['detection_rate']:>12.2%}{hinglish_results['detection_rate']:>12.2%}")
print(f"{'False Positive Rate':<30}{hindi_results['fpr']:>12.2%}{hinglish_results['fpr']:>12.2%}")
print(f"{'False Negative Rate':<30}{hindi_results['fnr']:>12.2%}{hinglish_results['fnr']:>12.2%}")

fpr_gap = abs(hindi_results['fpr'] - hinglish_results['fpr'])
fnr_gap = abs(hindi_results['fnr'] - hinglish_results['fnr'])

print(f"\nFAIRNESS ANALYSIS:")
print(f"FPR gap between languages: {fpr_gap:.2%}")
print(f"FNR gap between languages: {fnr_gap:.2%}")

if hindi_results['fpr'] > hinglish_results['fpr']:
    print(f"→ Hindi users wrongly blocked MORE than Hinglish")
elif hinglish_results['fpr'] > hindi_results['fpr']:
    print(f"→ Hinglish users wrongly blocked MORE than Hindi")
else:
    print(f"→ Equal false positive rates — no fairness gap")

total_samples = hindi_results['n_samples'] + hinglish_results['n_samples']
combined_acc  = (
    hindi_results['accuracy'] * hindi_results['n_samples'] +
    hinglish_results['accuracy'] * hinglish_results['n_samples']
) / total_samples

print(f"\nCOMPARISON WITH SRINIVASAN ET AL. (2026):")
print(f"Their accuracy:    99.70%")
print(f"Your accuracy:     {combined_acc:.2%}")
print(f"Difference:        {abs(99.70 - combined_acc * 100):.2f}%")

results_df = pd.DataFrame([hindi_results, hinglish_results])
results_df.to_csv('results_model1.csv', index=False)
print(f"\nResults saved to results_model1.csv")

# ============================================
# STEP 8: FIVE-TIER CONFIDENCE RISK SCORING
# ============================================
print("\n" + "="*60)
print("STEP 8: FIVE-TIER CONFIDENCE RISK SCORING")
print("="*60)

def assign_tier(conf):
    if conf >= 0.90:
        return 'CRITICAL'
    elif conf >= 0.70:
        return 'HIGH'
    elif conf >= 0.50:
        return 'MEDIUM'
    elif conf >= 0.30:
        return 'LOW'
    else:
        return 'SAFE'

def tier_distribution(name, vectors, n_samples):
    confidences = svm.predict_proba(vectors)[:, 1]
    tiers       = np.array([assign_tier(c) for c in confidences])

    critical = np.sum(tiers == 'CRITICAL')
    high     = np.sum(tiers == 'HIGH')
    medium   = np.sum(tiers == 'MEDIUM')
    low      = np.sum(tiers == 'LOW')
    safe     = np.sum(tiers == 'SAFE')

    print(f"\n{name} Tier Distribution ({n_samples} prompts):")
    print(f"  Critical (auto block):    {critical:>4} ({critical/n_samples:.1%}) → immediate block")
    print(f"  High     (urgent CSR):    {high:>4} ({high/n_samples:.1%}) → same day review")
    print(f"  Medium   (standard CSR):  {medium:>4} ({medium/n_samples:.1%}) → next day review")
    print(f"  Low      (log/monitor):   {low:>4} ({low/n_samples:.1%}) → batch review")
    print(f"  Safe     (auto allow):    {safe:>4} ({safe/n_samples:.1%}) → no action")
    print(f"\n  Auto-handled: {critical+safe} ({(critical+safe)/n_samples:.1%}) — no human needed")
    print(f"  Needs CSR:    {high+medium} ({(high+medium)/n_samples:.1%}) — human review required")

    return confidences, tiers

hi_confidences, hi_tiers = tier_distribution("HINDI",    hi_test_vectors, len(hi_test))
hg_confidences, hg_tiers = tier_distribution("HINGLISH", hg_test_vectors, len(hg_test))

# ============================================
# STEP 9: PROMPT INSPECTION
# ============================================
print("\n" + "="*60)
print("STEP 9: PROMPT INSPECTION — FLAGGED PROMPTS")
print("="*60)

def inspect_prompts(name, test_df, confidences, tiers):
    """
    Inspect false positives, false negatives by tier.
    Shows translations for Hindi prompts.
    """
    inspect_df = test_df.copy().reset_index(drop=True)
    inspect_df['attack_confidence'] = confidences
    inspect_df['tier']              = tiers
    inspect_df['true_label']        = inspect_df['label']

    has_translation = 'Translation' in inspect_df.columns

    # --- FALSE POSITIVES ---
    fp_df = inspect_df[
        (inspect_df['true_label'] == 0) &
        (inspect_df['tier'].isin(['CRITICAL', 'HIGH']))
    ].sort_values('attack_confidence', ascending=False)

    print(f"\n{'='*50}")
    print(f"FALSE POSITIVES — {name} ({len(fp_df)} prompts)")
    print(f"Innocent prompts wrongly blocked")
    print(f"{'='*50}")
    for _, row in fp_df.iterrows():
        print(f"Confidence: {row['attack_confidence']:.2%} | Tier: {row['tier']}")
        print(f"Prompt:     {row['text']}")
        if has_translation and pd.notna(row.get('Translation', None)):
            print(f"Translation:{row['Translation']}")
        print("-"*40)

    # --- FALSE NEGATIVES ---
    fn_df = inspect_df[
        (inspect_df['true_label'] == 1) &
        (inspect_df['tier'].isin(['LOW', 'SAFE']))
    ].sort_values('attack_confidence', ascending=True)

    print(f"\n{'='*50}")
    print(f"FALSE NEGATIVES — {name} ({len(fn_df)} prompts)")
    print(f"Real attacks that slipped through")
    print(f"{'='*50}")
    for _, row in fn_df.iterrows():
        print(f"Confidence: {row['attack_confidence']:.2%} | Tier: {row['tier']}")
        print(f"Prompt:     {row['text']}")
        if has_translation and pd.notna(row.get('Translation', None)):
            print(f"Translation:{row['Translation']}")
        print("-"*40)

    # --- PROMPTS BY TIER ---
    for tier in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'SAFE']:
        tier_df = inspect_df[
            inspect_df['tier'] == tier
        ].sort_values('attack_confidence', ascending=False)

        print(f"\n{'='*50}")
        print(f"{tier} TIER — {name} ({len(tier_df)} prompts)")
        print(f"{'='*50}")
        for _, row in tier_df.head(5).iterrows():
            actual  = "ATTACK"   if row['true_label'] == 1 else "INNOCENT"
            correct = "✓" if (
                (row['true_label'] == 1 and row['tier'] in ['CRITICAL', 'HIGH']) or
                (row['true_label'] == 0 and row['tier'] in ['LOW', 'SAFE'])
            ) else "✗ WRONG"
            print(f"Confidence: {row['attack_confidence']:.2%} | Actually: {actual} {correct}")
            print(f"Prompt:     {row['text']}")
            if has_translation and pd.notna(row.get('Translation', None)):
                print(f"Translation:{row['Translation']}")
            print("-"*40)

    filename = f"{name.lower()}_inspection.csv"
    inspect_df.to_csv(filename, index=False)
    print(f"\nFull {name} inspection saved to {filename}")

    return inspect_df

print("\nInspecting Hindi prompts...")
hi_inspect = inspect_prompts("HINDI",    hi_test, hi_confidences, hi_tiers)

print("\nInspecting Hinglish prompts...")
hg_inspect = inspect_prompts("HINGLISH", hg_test, hg_confidences, hg_tiers)

# ============================================
# STEP 10: FAIRNESS GAP SUMMARY
# ============================================
print("\n" + "="*60)
print("STEP 10: FAIRNESS GAP — HINDI VS HINGLISH")
print("="*60)

hi_fp  = len(hi_inspect[(hi_inspect['true_label'] == 0) & (hi_inspect['tier'].isin(['CRITICAL', 'HIGH']))])
hg_fp  = len(hg_inspect[(hg_inspect['true_label'] == 0) & (hg_inspect['tier'].isin(['CRITICAL', 'HIGH']))])
hi_fn  = len(hi_inspect[(hi_inspect['true_label'] == 1) & (hi_inspect['tier'].isin(['LOW', 'SAFE']))])
hg_fn  = len(hg_inspect[(hg_inspect['true_label'] == 1) & (hg_inspect['tier'].isin(['LOW', 'SAFE']))])
hi_csr = len(hi_inspect[hi_inspect['tier'].isin(['HIGH', 'MEDIUM'])])
hg_csr = len(hg_inspect[hg_inspect['tier'].isin(['HIGH', 'MEDIUM'])])

print(f"\n{'Metric':<40}{'Hindi':>10}{'Hinglish':>10}")
print("-"*60)
print(f"{'False Positives (wrongly blocked)':<40}{hi_fp:>10}{hg_fp:>10}")
print(f"{'False Negatives (attacks missed)':<40}{hi_fn:>10}{hg_fn:>10}")
print(f"{'CSR queue (High + Medium)':<40}{hi_csr:>10}{hg_csr:>10}")

print(f"\nKEY FINDINGS:")
if hg_fp > hi_fp:
    print(f"→ Hinglish has {hg_fp - hi_fp} more false positives than Hindi")
    print(f"  Innocent Hinglish users are wrongly blocked more often")
if hg_fn > hi_fn:
    print(f"→ Hinglish has {hg_fn - hi_fn} more attacks slipping through than Hindi")
    print(f"  Hinglish attack patterns are harder for the model to detect")
if hg_csr > hi_csr:
    print(f"→ Hinglish needs {hg_csr - hi_csr} more CSR reviews than Hindi")
    print(f"  Higher operational burden for Hinglish users")

# ============================================
# EXTERNAL VALIDATION: UPADHAYAY STEALTH DATASETS
# ============================================
print("\n" + "="*60)
print("EXTERNAL VALIDATION: UPADHAYAY STEALTH DATASETS")
print("="*60)

print("\nLoading local datasets...")

def load_dataset(filename):
    try:
        df_ext = pd.read_csv(filename)
        print(f"  ✓ {filename}")
        print(f"    Rows:    {len(df_ext)}")
        print(f"    Columns: {df_ext.columns.tolist()}")
        df_ext['text']  = df_ext['hinglish'].astype(str)
        df_ext['label'] = df_ext['expected'].map(
            {'BLOCK': 1, 'ALLOW': 0, 'block': 1, 'allow': 0}
        ).fillna(0).astype(int)
        print(f"    Labels:  {df_ext['label'].value_counts().to_dict()}")
        return df_ext[['text', 'label', 'category']].reset_index(drop=True)
    except FileNotFoundError:
        print(f"  ✗ {filename} not found")
        return None
    except KeyError as e:
        print(f"  ✗ Missing column in {filename}: {e}")
        return None

stealth_250 = load_dataset('hinglish-stealth-250.csv')
stealth_110 = load_dataset('hinglish-stealth-110-heldout.csv')
clean_500   = load_dataset('hinglish-clean-500.csv')

def evaluate_external(name, df_ext, upadhayay_detection=None, upadhayay_fpr=None):
    if df_ext is None:
        print(f"\nSkipping {name} — not loaded")
        return None

    print(f"\n{'='*50}")
    print(f"DATASET: {name} ({len(df_ext)} prompts)")
    print(f"{'='*50}")

    print("  Encoding with MiniLM...")
    vectors = embedder.encode(
        df_ext['text'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )

    predictions   = svm.predict(vectors)
    probabilities = svm.predict_proba(vectors)[:, 1]
    true_labels   = df_ext['label'].values
    categories    = df_ext['category'].values
    texts         = df_ext['text'].values
    total         = len(true_labels)

    tiers    = np.array([assign_tier(c) for c in probabilities])
    critical = int(np.sum(tiers == 'CRITICAL'))
    high     = int(np.sum(tiers == 'HIGH'))
    medium   = int(np.sum(tiers == 'MEDIUM'))
    low      = int(np.sum(tiers == 'LOW'))
    safe     = int(np.sum(tiers == 'SAFE'))

    unique_labels = np.unique(true_labels)

    # All attacks
    if len(unique_labels) == 1 and unique_labels[0] == 1:
        tp             = int(np.sum((predictions == 1) & (true_labels == 1)))
        fn             = int(np.sum((predictions == 0) & (true_labels == 1)))
        detection_rate = tp / total if total > 0 else 0
        fnr            = fn / total if total > 0 else 0

        print(f"\n  All prompts are attacks (label=1)")
        print(f"  Detection Rate:      {detection_rate:.2%}  (attacks caught)")
        print(f"  False Negative Rate: {fnr:.2%}  (attacks missed)")
        print(f"  Attacks caught:      {tp}")
        print(f"  Attacks missed:      {fn}")
        print(f"\n  Tier Distribution:")
        print(f"    Critical (auto block):  {critical:>4} ({critical/total:.1%})")
        print(f"    High     (urgent CSR):  {high:>4} ({high/total:.1%})")
        print(f"    Medium   (CSR review):  {medium:>4} ({medium/total:.1%})")
        print(f"    Low      (monitor):     {low:>4} ({low/total:.1%})")
        print(f"    Safe     (auto allow):  {safe:>4} ({safe/total:.1%})")

        if upadhayay_detection is not None:
            gap = upadhayay_detection - detection_rate
            print(f"\n  vs Upadhayay (2026):")
            print(f"    Your Detection Rate:       {detection_rate:.2%}")
            print(f"    Upadhayay Detection Rate:  {upadhayay_detection:.2%}")
            print(f"    Gap:                       {gap:+.2%}")
            if gap > 0:
                print(f"    → Upadhayay catches {gap:.2%} more stealth attacks")
                print(f"      Gap = value of their rule engine + contextual guard")
            else:
                print(f"    → Your model catches more attacks than Upadhayay")

        fn_mask       = (predictions == 0) & (true_labels == 1)
        fn_texts      = texts[fn_mask]
        fn_confs      = probabilities[fn_mask]
        fn_tiers      = tiers[fn_mask]
        fn_categories = categories[fn_mask]

        if len(fn_texts) > 0:
            print(f"\n  Missed attacks by category:")
            unique_cats, cat_counts = np.unique(fn_categories, return_counts=True)
            for cat, count in sorted(
                [(str(c), int(n)) for c, n in zip(unique_cats, cat_counts)],
                key=lambda x: -x[1]
            ):
                print(f"    {cat}: {count} missed")

            sort_idx = np.argsort(fn_confs)
            print(f"\n  Sample missed attacks (hardest to detect):")
            for i in sort_idx[:5]:
                print(f"    Confidence: {fn_confs[i]:.2%} | "
                      f"Tier: {fn_tiers[i]} | "
                      f"Category: {fn_categories[i]}")
                print(f"    Prompt: {fn_texts[i]}")
                print(f"    {'-'*40}")
        else:
            print(f"\n  No missed attacks — model caught all prompts")

        return {
            'dataset': name, 'total': total,
            'detection_rate': detection_rate, 'fnr': fnr,
            'fpr': None, 'accuracy': None,
            'tp': tp, 'fn': fn, 'fp': None, 'tn': None,
            'critical': critical, 'high': high,
            'medium': medium, 'low': low, 'safe': safe,
        }

    # All innocent
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        fp  = int(np.sum((predictions == 1) & (true_labels == 0)))
        tn  = int(np.sum((predictions == 0) & (true_labels == 0)))
        fpr = fp / total if total > 0 else 0

        print(f"\n  All prompts are innocent (label=0)")
        print(f"  False Positive Rate: {fpr:.2%}  (innocent wrongly blocked)")
        print(f"  False Positives:     {fp}  (wrongly blocked)")
        print(f"  True Negatives:      {tn}  (correctly allowed)")
        print(f"\n  Tier Distribution:")
        print(f"    Critical (auto block):  {critical:>4} ({critical/total:.1%})")
        print(f"    High     (urgent CSR):  {high:>4} ({high/total:.1%})")
        print(f"    Medium   (CSR review):  {medium:>4} ({medium/total:.1%})")
        print(f"    Low      (monitor):     {low:>4} ({low/total:.1%})")
        print(f"    Safe     (auto allow):  {safe:>4} ({safe/total:.1%})")

        if upadhayay_fpr is not None:
            gap = fpr - upadhayay_fpr
            print(f"\n  vs Upadhayay (2026):")
            print(f"    Your FPR:       {fpr:.2%}")
            print(f"    Upadhayay FPR:  {upadhayay_fpr:.2%}")
            print(f"    Gap:            {gap:+.2%}")
            if gap > 0:
                print(f"    → Your model wrongly blocks {gap:.2%} more innocent prompts")
            else:
                print(f"    → Your model blocks fewer innocent prompts than Upadhayay")

        fp_mask       = (predictions == 1) & (true_labels == 0)
        fp_texts      = texts[fp_mask]
        fp_confs      = probabilities[fp_mask]
        fp_tiers      = tiers[fp_mask]
        fp_categories = categories[fp_mask]

        if len(fp_texts) > 0:
            print(f"\n  Wrongly blocked by category:")
            unique_cats, cat_counts = np.unique(fp_categories, return_counts=True)
            for cat, count in sorted(
                [(str(c), int(n)) for c, n in zip(unique_cats, cat_counts)],
                key=lambda x: -x[1]
            ):
                print(f"    {cat}: {count} wrongly blocked")

            sort_idx = np.argsort(fp_confs)[::-1]
            print(f"\n  Sample wrongly blocked innocent prompts:")
            for i in sort_idx[:5]:
                print(f"    Confidence: {fp_confs[i]:.2%} | "
                      f"Tier: {fp_tiers[i]} | "
                      f"Category: {fp_categories[i]}")
                print(f"    Prompt: {fp_texts[i]}")
                print(f"    {'-'*40}")
        else:
            print(f"\n  No false positives — all innocent prompts correctly allowed")

        return {
            'dataset': name, 'total': total,
            'fpr': fpr, 'detection_rate': None,
            'fnr': None, 'accuracy': None,
            'fp': fp, 'tn': tn, 'tp': None, 'fn': None,
            'critical': critical, 'high': high,
            'medium': medium, 'low': low, 'safe': safe,
        }

    print(f"  WARNING: Unexpected label distribution {unique_labels}")
    return None

print("\n" + "="*60)
print("RUNNING EXTERNAL EVALUATION")
print("="*60)

results_250 = evaluate_external(
    'Stealth 250 (public benchmark)', stealth_250, upadhayay_detection=0.984)
results_110 = evaluate_external(
    'Stealth 110 (blind heldout)', stealth_110, upadhayay_detection=0.9727)
results_500 = evaluate_external(
    'Clean 500 (benign Hinglish)', clean_500, upadhayay_fpr=0.006)

print("\n" + "="*60)
print("FINAL COMPARISON SUMMARY")
print("="*60)

print(f"\n{'Dataset':<32}{'Metric':<22}{'Your Model':>12}"
      f"{'Upadhayay':>12}{'Gap':>10}")
print("-"*90)

if results_250 and results_250.get('detection_rate') is not None:
    dr  = results_250['detection_rate']
    gap = 0.984 - dr
    print(f"{'Stealth 250':<32}{'Detection Rate':<22}"
          f"{dr:>12.2%}{0.984:>12.2%}{gap:>+10.2%}")

if results_110 and results_110.get('detection_rate') is not None:
    dr  = results_110['detection_rate']
    gap = 0.9727 - dr
    print(f"{'Stealth 110':<32}{'Detection Rate':<22}"
          f"{dr:>12.2%}{0.9727:>12.2%}{gap:>+10.2%}")

if results_500 and results_500.get('fpr') is not None:
    fpr = results_500['fpr']
    gap = fpr - 0.006
    print(f"{'Clean 500':<32}{'False Positive Rate':<22}"
          f"{fpr:>12.2%}{0.006:>12.2%}{gap:>+10.2%}")

if not any([
    results_250 and results_250.get('detection_rate') is not None,
    results_110 and results_110.get('detection_rate') is not None,
    results_500 and results_500.get('fpr') is not None,
]):
    print("  No results to display — check datasets loaded correctly")

print(f"\nKEY INSIGHT:")
print(f"  Your model  = MiniLM + SVM only (Layer 4 equivalent)")
print(f"  Upadhayay   = 5 layers: normalize + rules + contextual")
print(f"                guard + SVM + decision engine")
print(f"  Detection gap = value their extra layers add on top of SVM")
print(f"  This motivates adding rule-based or contextual layers")
print(f"  as future work to close the gap.")

all_results = [r for r in [results_250, results_110, results_500] if r]
if all_results:
    ext_df = pd.DataFrame(all_results)
    ext_df.to_csv('results_external_validation.csv', index=False)
    print(f"\nResults saved to results_external_validation.csv")