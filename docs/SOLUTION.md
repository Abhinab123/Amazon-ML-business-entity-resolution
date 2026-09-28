# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Magnum  
**Team Members:** Abhinab Bezbaruah, Aman Singh Negi, Arjun Baidya  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

The pipeline works in four stages:
1. **Blocking.** Country-constrained, per-source, multi-pass retrieval: word / char / address
   TF-IDF, multilingual embeddings, exact keys, and a **fine-tuned bi-encoder retrieval pass**. It
   is followed by **sibling expansion**: each S1 entity's confident candidates query the pool for
   their own near-duplicates. Candidate recall is 0.9952.
2. **Stage-1 LightGBM** pair classifier.
3. **Four fine-tuned multilingual cross-encoders**: MiniLM-L12 (Apache-2.0), XLM-R base, XLM-R base
   with a hard-example epoch, and XLM-R large (MIT). They read both records together.
4. **Stage-2 LightGBM** that re-scores each pair from its neighbours: rival owners of the pool
   record, the S1's confident siblings, and whether the record has near-duplicates at all.

The final stage-2 keeps only features that transfer to an unseen country. They were chosen by
training on one training country and scoring the other, because France appears only in test.
Out-of-fold macro F0.5 is **0.9907** on 600k training entities (v15). The final submission averages the v12, v13 and v15 stage-2 models: public leaderboard **0.984833**. Leaderboard scores of every
version are in `VERSION_LOG.md`.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the training split (2.21M S1 entities, 10.32M S2+S3 records):

| Finding | Value | Consequence |
|---|---|---|
| Singleton rate | 5.58% (same for US and India) | recall matters as much as avoiding false merges |
| Matches per S1 | mean 3.46 (≈1.7 S2 + 1.8 S3) | retrieve per source; the true matches form a *cluster* |
| Pool records owned by some S1 | 74%: **26% are distractors** built from S1 names | the main source of false merges (85% of our false-positive pairs) |
| Pool records with > 1 owner | **0** | one-owner constraint, used as learned rival-owner features |
| Matched pairs sharing the country label | 100% | country used only as an equality key |
| Test | 1.73M S1: India 810k, US 663k, **France 259k (unseen in train)** | no country-specific logic anywhere |

Noise seen in matched groups:
- **Names:**
  - legal-suffix changes (Pvt Ltd / Private Limited / LLC; SARL / SAS / EURL in test);
  - typos and reordering;
  - website / hashtag forms;
  - garbled names paired with a correct address;
  - **native-script names** (Devanagari, Malayalam, Kannada, Bengali) for Indian businesses;
  - "first token + legal + generic word" variants ("Roopaya Storage Limited" →
    "Roopaya Limited Services").
- **Addresses:**
  - reordered components;
  - abbreviations (St, Rd; French `R.`, `N°`);
  - "10ST" ordinals, leading zeros, `NULL`;
  - added unit / plot numbers;
  - state codes vs names;
  - landmarks;
  - truncation.
- Test-only France pattern: **same name at a neighbouring house number** (5 vs 7 rue …).

### 2.2 Solution Strategy
**Approach Type:** Hybrid. Two-round blocking with sibling expansion, a two-stage gradient-boosted
classifier stacked with two fine-tuned cross-encoders, and a macro-F0.5 decision layer.

**Core Innovations:**
1. **Cluster-aware blocking and features.**
   - The true matches of an entity are near-duplicates of each other.
   - *Sibling expansion* uses an entity's confident candidates as extra queries. It raised train
     blocking recall from 0.9696 to **0.9813**.
   - *Sibling features* score each candidate against the entity's confident candidates.
2. **One-owner rule as learned evidence.**
   - Reverse-rank features in stage-1: this S1's rank among all S1s that retrieved the record.
   - The best *rival* S1's stage-1 probability, in stage-2.
3. **Distractor detection by "twins."**
   - Genuine records almost always have a near-duplicate elsewhere in the pool (the other source's
     copy); synthetic distractors usually do not.
   - The pool-twin similarity alone separates them with AUC 0.79. It lifted singleton F0.5
     0.988 → 0.992.
4. **Fine-tuned multilingual cross-encoders.**
   - They read raw text in any script: `बाबा पावर प्राइवेट लिमिटेड` ↔ "Baba Power Private Limited"
     gets logit +9.0.
   - Trained on train S1 entities disjoint from the stage-2 sample, so stacking is leak-free.
   - They are the largest single gain: +0.009 CV.
5. **IDF-weighted name overlap.** Separates a shared *rare* token from a shared generic one
   ("Sai", "Shree"): +0.010 CV.

---

## 3. Candidate Generation (Blocking)

Every S1 record is compared only with pool records carrying the same country string (an equality
key; France is handled exactly like US and India). Retrieval is done separately from S2 and from
S3.

- **Blocking keys used:**

  | Pass | Representation | K / source |
  |---|---|---|
  | word | TF-IDF uni+bi-grams of name core + address core (df ≤ 3000), exact sparse cosine | 10 |
  | emb | MiniLM embedding of raw "name, address", exact GPU cosine | 3 |
  | char | char-3-gram TF-IDF of compact name (df ≤ 20000) | 3 |
  | addr-nonLatin | address TF-IDF against pool records whose **name is non-Latin script** | 5 |
  | keys | (house number, first name token), (metaphone, city guess), groups ≤ 30 | all |
  | **sibling** | word TF-IDF neighbours of the S1's confident candidates. Round 1: top-3 by cheap score; round 2: p1 ≥ 0.5 from the round-1 model, top 5 | 3 |

- **Candidate pairs generated:**
  - train 85.0M (38.5 per S1), recall **0.9813**, reduction ratio 0.9999963;
  - test **65.9M** (India 44.0, US 31.8, France 35.4 per S1).
- **How we ensured true matches were not lost:**
  - K tuned on a 100k-S1 sample against the full pool; the union plateaus near 0.98 even at 287
    candidates per S1, so K sits at the knee.
  - Wider K was tested (0.9796 recall at 65 per S1) and *hurt* stage-1 F0.5, because it adds more
    noise than recall.
  - Targeted passes were designed from miss analysis:
    - 46% of misses had non-Latin names → the address pass against non-Latin-name records;
    - the rest were far from S1 but near its other matches → sibling expansion.

---

## 4. Matching Model

**Features used:**
- **Name features:**
  - rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler on the normalised name core;
  - compact-name ratios;
  - metaphone and **consonant-skeleton** similarity (transliteration-robust);
  - DBA-variant maximum;
  - acronym, first/last token, legal-suffix state (both missing / equal / one missing / conflict);
  - **IDF-weighted overlap**: covered weight share for each side, IDF of the rarest shared,
    S1-only and candidate-only token, number of candidate-only tokens.
- **Address features:**
  - token-set / sort / ratio / partial on the address core and street words;
  - house-number Jaccard, conflict and containment;
  - primary-number state;
  - postcode state and 3-digit prefix;
  - city agreement;
  - landmarks;
  - empty-address flag.
- **Other:**
  - MiniLM embedding cosines (full text and name only);
  - word / char / address TF-IDF cosines;
  - **context**: rank and gap within the S1 list, reverse rank / gap among all S1s retrieving the
    record, list sizes, mutual-best flag, name-core frequency (chains);
  - per-pass retrieval ranks;
  - **pool-twin similarities**;
  - `same_country` (never the country value itself).
- **Stage-2 features:**
  - own stage-1 probability p1;
  - S1-list max / second / sum / count > 0.5 and own rank;
  - within-source rank;
  - **best rival owner's p1**, own rank among owners, margin to the rival;
  - **sibling similarities** (word / address TF-IDF to confident siblings, plain and
    confidence-weighted);
  - **cross-encoder logits** (MiniLM, XLM-R) for pairs with p1 ≥ 0.01.

**Model type:**
- **Stage-1:** LightGBM (63 leaves, lr 0.05, early stopping) on a 600k-S1 random sample of train
  (23M pairs); 5-fold GroupKFold by S1 → out-of-fold p1; isotonic calibration. p1 for the other
  1.6M train S1 comes from the final model (out-of-sample).
- **Cross-encoders:** each backbone gets a 1-logit head.
  - Fine-tuned for one epoch (AdamW, bf16, max length 96) on all positives plus 6 hard and
    2 random negatives of 200k train S1 **disjoint from the stage-2 sample**.
  - Holdout pair AUC on 10k unseen S1: MiniLM 0.99972, XLM-R **0.99986**.
- **Stage-2:** LightGBM on the features above; 5-fold GroupKFold OOF; isotonic. Feature gain:
  XLM-R 0.67, MiniLM 0.17, p1 0.13, rival margin and siblings the rest.

**Threshold selection method:**
- A grid on OOF macro F0.5, using the exact challenge formula over all sampled entities
  (singletons included).
- Candidates: a global threshold 0.30–0.90, and per-entity expected-F0.5 subset selection
  (temperature × p_min grid), each with one-owner damping α ∈ {0, 0.3, 1}.
- **Selected (v7): threshold 0.65, α = 0.** Expected-F0.5 selection was tied within 0.0001.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9905** out-of-fold for the final model (v13; singletons 0.997,
  non-singletons 0.990).
  - Held-out-country F0.5 (stage-2 trained on the other training country): India 0.9911, US 0.9894.
  - Public leaderboard: **0.984833** (final blend of v12 / v13 / v15), up from 0.9778 (v9).

  | Step | CV | LB |
  |---|---|---|
  | baseline blocking + LightGBM | 0.9597 | – |
  | + IDF name features | 0.9698 | – |
  | + non-Latin address pass, stage-2 stacking | 0.9727 | 0.957 |
  | + French normalisation fixes | ≈0.9727 | 0.958 |
  | + sibling blocking / features, MiniLM cross-encoder | 0.9820 | 0.973 |
  | + model siblings, pool twins | 0.9835 | – |
  | + XLM-R cross-encoder | 0.9851 | 0.9768 |
  | + hard-example XLM-R epoch (S1 outside the stage-2 sample) | 0.9855 | – |
  | + 3-seed stage-2 ensemble (v9) | 0.9856 | 0.9778 |
  | + bi-encoder retrieval pass, MiniLM CE only (v10m) | 0.9881 | 0.9769 |
  | + XLM-R base / hard-example / large; unseen-country stage-2 (no bi-encoder / density features, threshold 0.8) (v13) | 0.9905 | 0.98474 |
  | + stage-2 min_child_samples 500 (v15) | 0.9907 | 0.984826 |
  | average of the v12 / v13 / v15 stage-2 probabilities, threshold 0.8 (**final**) | – | **0.984833** |

- **Generalisation to an unseen country.** Leave-country-out, stage-1, training on one country
  only: held-out US 0.955–0.961, held-out India 0.825–0.840.
  - That drop motivated a pipeline without country-specific logic, text-level cross-encoders, and
    France-agnostic normalisation fixes (`N°`, `R.` = rue, `St` = Saint), which changed 24% of
    French predictions.
  - On test, France's statistics match the training countries: 3.25–3.30 matches per S1, and
    5.7–6.0% empty predictions against a 5.6% singleton rate.
- **Common false positives (wrong merges):**
  - **distractors built from an S1 name** ("Veis" → "Veis Incorporated", "DENET LLC" with no
    address): 85% of false-positive pairs before the twin features;
  - records of a different S1 with a near-identical name (chains; common Indian names);
  - in France, the same name at a neighbouring house number.
- **Common false negatives (missed matches):**
  - native-script names with truncated addresses, now largely recovered by the non-Latin address
    pass, sibling expansion and cross-encoders;
  - "first token + legal + generic word" variants with partial addresses when a rival S1 shares
    the first token.
- **What did not help:**
  - self-training LightGBM on unseen-country pseudo-labels (≤ +0.004, unstable);
  - adapting XLM-R on confident French test pseudo-pairs: leaderboard slightly below v7's 0.9768 (dropped;
    the final model uses no test-derived training signal);
  - trusting the bi-encoder's own score in stage-2 (v10m): CV +0.0025 but leaderboard −0.0009. It
    over-matched France (fewer empty predictions than the training singleton rate). The
    leave-country-out check confirmed that these features, and the candidate-density features,
    transfer badly, so the final model drops them.
  - one-owner strength α, sibling-feature removal, base-score removal: neutral in the
    leave-country-out check;
  - per-entity sample weighting in stage-2 (1/√n): 0.9854 vs 0.9855;
  - wider K;
  - dropping embedding / retrieval features once IDF features existed.

---

## 6. Conclusion

The largest gains came from treating matching as a **cluster problem**:
- the one-owner constraint and rival scores;
- sibling expansion and sibling features;
- pool twins;
- and from **cross-encoders that read raw multilingual text**, which close most of the gap left
  by string metrics on transliterated and garbled records.

What remains is mostly transfer to the unseen country and a ~2% blocking ceiling.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: the full reproduction runs from `student_resource/`:
```bash
bash code/business_entity_resolution/reproduce.sh
```

It runs:
1. `run_pipeline.py` round 1 (cheap-score siblings; trains the MiniLM cross-encoder);
2. round 2 (siblings from the round-1 model, pool twins; train + test);
3. `train_cross_encoder.py` (XLM-R);
4. `ce_score_pairs.py`;
5. `stage2_refit.py` → `output/matching_results.tsv`, `output/candidate_pairs.tsv`;
6. the official validator.

| Module | Role |
|---|---|
| `normalize.py` | normalisation |
| `blocking.py` | passes, sibling expansion, pool twins |
| `features.py` | pair and context features |
| `model.py` | LightGBM OOF + isotonic |
| `stage2.py` / `siblings.py` | stacking features |
| `cross_encoder.py` | fine-tuning and scoring |
| `decide.py` / `cv.py` | decision layer and tuning |
| `run_pipeline.py` | one round, end to end |

Diagnostics used during development: `eda.py`, `tune_blocking.py`, `analyze.py`, `experiment.py`.

**Models and licences:**
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: Apache-2.0, 118M;
- `FacebookAI/xlm-roberta-base`: MIT, 278M;
- LightGBM: MIT.

**No external data, APIs, or geocoding used.** All statistics are fitted on the split's own
unlabelled text; all supervised models use train labels only.

### B. Additional Results

Blocking K tuning (100k train S1 vs the full pool):

| Configuration | Recall | Candidates / S1 |
|---|---|---|
| word TF-IDF only, K = 5 / 10 / 40 | 0.924 / 0.949 / 0.970 | 10 / 20 / 80 |
| embedding only, K = 5 / 10 / 40 | 0.645 / 0.681 / 0.734 | 10 / 20 / 80 |
| all passes, K = 5 / 15 / 40 each | 0.960 / 0.976 / 0.983 | 36 / 107 / 287 |
| chosen direct passes | 0.9696 | 35.4 |
| + sibling expansion (cheap-score / model siblings) | 0.9742 / 0.9813 | 37.0 / 38.5 |
| + fine-tuned bi-encoder pass `f`, K=5 per source | **0.9952** | 43.3 |

Leave-country-out feature-group ablation (v1). Each value is the change in held-out F0.5 when the
group is removed:

| Group | India held out | US held out |
|---|---|---|
| embeddings | +0.017 | +0.005 |
| retrieval ranks | +0.008 | +0.006 |
| legal suffix | +0.015 | −0.009 |
| context | −0.023 | +0.002 |
| name frequency | −0.008 | −0.002 |
