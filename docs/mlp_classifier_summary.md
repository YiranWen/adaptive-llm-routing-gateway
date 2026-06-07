# MLP Classifier Summary

This document summarizes the final local-failure classifier evaluation used for
the report-facing MLP Classifier baseline.

## Why the Original MLP Overfit

The original MLP classifier used the full 902D context:

```text
896D Qwen prompt feature + 6D system feature
```

However, the classifier target is:

```text
local_failure = score_local < 0.7
```

This target is mostly prompt-dependent. It should not change because CPU, memory, battery, budget, or latency changes. Therefore, the 6D system features are useful for the Utility Prediction Router, but they act mostly like noise for local_failure classification.

The original MLP also had a high-dimensional input and a small imbalanced dataset:

```text
train prompts: 512
validation prompts: 128
positive validation examples: 15
```

This made overfitting likely. The original MLP reached perfect training performance but weaker validation positive-class metrics.

## Clean Retraining Strategy

The MLP Classifier uses:

```text
896D Qwen prompt features only
```

It removes the 6D system features because they are not causally related to local_failure.

The model is:

```text
StandardScaler -> Logistic Regression
```

Configuration:

```text
C = 3.0
solver = liblinear
threshold = 0.16
```

The threshold was selected using an internal development split from the training set, not the validation set:

```text
75% train subset for fitting
25% internal dev subset for threshold selection
original validation set for final reporting
```

This avoids directly tuning the decision threshold on the final validation set.

## Validation Confusion Matrix

Positive class:

```text
local_failure = 1
```

Confusion matrix:

```text
                         Pred No Failure    Pred Failure
True No Failure                 90              23
True Failure                     7               8
```

Values:

```text
TN = 90
FP = 23
FN = 7
TP = 8
```

## Validation Metrics

| Metric | Original MLP | MLP Classifier |
| --- | ---: | ---: |
| Accuracy | 0.7969 | 0.7656 |
| Precision | 0.2381 | 0.2581 |
| Recall / TPR | 0.3333 | 0.5333 |
| F1 | 0.2778 | 0.3478 |
| FPR | 0.1416 | 0.2035 |
| PR-AUC | 0.2690 | 0.3400 |
| ROC-AUC | 0.6577 | 0.7009 |

## Interpretation

The MLP Classifier has slightly lower overall accuracy, but it is better at detecting the important positive class:

```text
local_failure = 1
```

Recall improves from 0.3333 to 0.5333, meaning it catches more true local failures.

F1 improves from 0.2778 to 0.3478.

PR-AUC and ROC-AUC also improve, which indicates better ranking quality.

The tradeoff is a higher false positive rate:

```text
FPR increases from 0.1416 to 0.2035
```

This means the MLP Classifier escalates more non-failure prompts than the original MLP. For an LLM router, this may be acceptable if missing true local failures is more harmful than extra cloud escalation.

## Honest Limitation

The MLP Classifier is cleaner, but the confusion matrix is still not excellent in an absolute sense. The validation set contains only 15 positive examples, so precision, recall, and F1 are sensitive to a small number of predictions.

The main honest conclusion is:

```text
Prompt-only regularized classification improves local_failure detection, but the small imbalanced dataset limits classifier quality.
```

This supports the project argument that local_failure classification alone is not the best final formulation. Utility-aware routing remains the stronger system-level approach.
