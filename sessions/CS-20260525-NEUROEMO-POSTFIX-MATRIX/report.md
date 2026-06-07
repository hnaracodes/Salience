# NeuroEmo Postfix Matrix Report

**Session ID:** `CS-20260525-NEUROEMO-POSTFIX-MATRIX`  
**Agent signature:** `Historical Cursor coding agent`  
**Context contract:** Future agents may cite this session ID when asking about the first post-fix NeuroEmo training matrix, metrics comparison, and new training-contract validation.  
**Metadata normalized by:** `GPT-5.5, 2026-06-07`

## Purpose

This note summarizes the first post-fix NeuroEmo training matrix after the non-surface-native items in `.cursor/plans/fix-neuroemo-training_029d2a6f.plan.md` were implemented.

It compares the new saved metrics against the earlier baselines documented in `coding-sessions/2026-05-25-neuroemo-training-modeling-upgrade.md`, records the focused pytest validation that was added for the new training contract, and explains the most plausible reasons accuracy improved.

## Baseline vs Postfix

Earlier documented baselines from the modeling-upgrade note:

- 5-class MLP: `mean_accuracy ~= 0.40`
- Binary valence MLP: `mean_accuracy ~= 0.64`
- Combined specialist model: `mean_accuracy ~= 0.40`

Current postfix matrix from `scout_data/neuroemo/models/2026-05-25_postfix_matrix/`:

| Model | Mean Accuracy | Mean Balanced Accuracy | Mean Macro F1 | Pooled Log Loss | Comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| `logistic_saga_5class_10tr` | `0.445000` | `0.445000` | `0.433494` | `2.279231` | Best current 5-class run |
| `mlp_5class_10tr` | `0.435000` | `0.435000` | `0.425113` | `1.632369` | Up vs prior ~`0.40` 5-class MLP baseline |
| `sgd_logistic_5class_10tr` | `0.415000` | `0.415000` | `0.409784` | `20.922971` | Better than old ~`0.40`, but poor probability quality |
| `linear_svc_5class_10tr` | `0.362500` | `0.362500` | `0.353384` | `1.625795` | Useful linear baseline, not top performer |
| `specialist_none_5class_10tr` | `0.380000` | `0.380000` | `0.369733` | `2.151962` | Below earlier ~`0.40` specialist baseline |
| `specialist_sigmoid_5class_10tr` | `0.350000` | `0.350000` | `0.318095` | `1.492396` | Better log loss than uncalibrated specialists, worse top-1 accuracy |
| `mlp_valence_binary_10tr` | `0.659375` | `0.659375` | `0.655723` | `0.710945` | Slightly up vs prior ~`0.64` binary valence baseline |

## Headline Takeaways

- The strongest 5-class result in the postfix matrix is now `logistic_saga_5class_10tr` at `0.445`.
- The 5-class MLP improved from roughly `0.40` to `0.435`.
- The binary valence MLP improved from roughly `0.64` to `0.659375`.
- Specialist models did not improve in final argmax accuracy, even though sigmoid calibration improved log-loss behavior.

## Focused Validation

Focused tests were added to `tests/test_neuroemo_training.py` and executed with:

```bash
python -m pytest tests/test_neuroemo_training.py
```

Result:

```text
14 passed in 3.36s
```

These tests now verify:

- manifest vertex-order rejection for unsupported orderings;
- manifest SHA rejection for `vertex_regions.csv` drift;
- ROI-manifest vs NPZ mismatch rejection before fit;
- re-windowing is blocked by default and only allowed with explicit opt-in;
- saved postfix metrics consistently record:
  - `fsaverage5`;
  - canonical `lh_then_rh_fsaverage5`;
  - source alias `facebook/tribev2_lh_then_rh`;
  - projected Schaefer provenance;
  - 10-TR contiguous temporal aggregation.

## Most Likely Causes Of The Accuracy Increase

The increase is most plausibly a **de-confounding gain**, not a fundamentally new representation gain.

### 1. Stronger training-contract enforcement

This is the most likely cross-model contributor.

The current pipeline now enforces:

- canonical vertex order and hemisphere order in `scout_core/parcellation.py`;
- manifest/hash-compatible ROI loading in `scout_core/roi_features.py`;
- explicit temporal provenance in `scripts/prepare_neuroemo_tribev2.py`;
- rejection of accidental double-windowing and missing temporal metadata in `scripts/train_neuroemo_emotion_model.py`.

Why this likely helped:

- earlier experiments could train on slightly mismatched or ambiguously prepared inputs without failing loudly;
- the postfix matrix removes more silent contract drift, so the model is more likely to be training on the intended data representation.

Evidence level: direct for the contract enforcement itself; inferential for the amount of accuracy gain it caused.

### 2. Better linear solver choice

This is the clearest directly supported reason for the best current 5-class result.

Under the same postfix ROI/temporal contract:

- `logistic_saga_5class_10tr` = `0.445000`
- `sgd_logistic_5class_10tr` = `0.415000`
- `linear_svc_5class_10tr` = `0.362500`

Why this likely helped:

- `logistic_saga` is simply a stronger optimizer/baseline than the SGD setup for this feature space and sample count.

Evidence level: direct.

### 3. Same MLP recipe on a hardened data path

The current 5-class MLP is still using the same general successful recipe family from the earlier note:

- 10-TR windows;
- Schaefer ROI features;
- `mean,std,mean_abs`;
- small hidden layer;
- regularized training.

Because the model family is not radically different, the modest lift from ~`0.40` to `0.435` is more likely explained by the cleaned-up training path than by a major architectural breakthrough.

Evidence level: mixed.

## What Did Not Cause The Gain

- The current matrix still uses the **projected** Schaefer artifact, not the deferred surface-native replacement.
- The current postfix metrics still record `preprocess_bold = false`.
- Specialist sigmoid calibration did **not** improve final 5-class top-1 accuracy.

This means the observed gain did not come from surface-native atlas adoption, heavy preprocessing, or the specialist-calibration path.

## Caveats

- The uplift is real but still modest. This is an improvement, not a solved discrete-emotion problem.
- The projected Schaefer atlas remains a quality ceiling.
- `sgd_logistic` still looks unreliable as a probability model because its log loss is extremely poor.
- Specialists show a useful calibration tradeoff: score behavior can improve while final multiclass argmax accuracy gets worse.

## Bottom Line

The post-fix matrix is better interpreted as:

1. the intended NeuroEmo training contract is now enforced and auditable;
2. the strongest current 5-class result comes from a better linear baseline (`logistic_saga`);
3. the MLP and binary valence runs improved modestly after the fixes;
4. the remaining ceiling is still likely tied to atlas quality and the intrinsic difficulty of the 5-class task.
