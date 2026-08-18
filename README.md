# DepositIQ - Term Deposit Subscription Predictor

BITS Pilani WILP · M.Tech (AIML/DSE) · Machine Learning Assignment 2
**Prarthana Naik · 2025AC05312**

## a. Problem statement

A Portuguese bank runs phone-call marketing campaigns to sell term deposits.
Calling every client is expensive, so the bank wants to predict — *before* a
call — which clients are likely to subscribe to a term deposit, so telemarketers
can prioritise the clients most worth calling. This is a binary classification
problem: predict `y` (subscribed: yes/no) from a client's profile and campaign
history.

## b. Dataset description

**UCI Bank Marketing** dataset (id 222, `bank-full.csv`).

| | |
|---|---|
| Source | https://archive.ics.uci.edu/dataset/222/bank+marketing |
| Instances | 45,211 |
| Raw input features | 16 |
| Features used for modelling | 15 (see note below) |
| Target | `y` — binary (`yes` / `no`), subscribed to a term deposit |
| Missing values | none |
| Class balance | 88.30% `no` / 11.70% `yes` (**imbalanced**) |

**Features:** `age`, `job`, `marital`, `education`, `default`, `balance`,
`housing`, `loan`, `contact`, `day`, `month`, `campaign`, `pdays`, `previous`,
`poutcome`.

**Dropped feature — `duration`:** the original 17th input, `duration` (length
of the last call, in seconds), was excluded before training. Per the UCI
documentation this value is only known *after* a call ends, and it is highly
predictive by construction (`duration = 0` implies `y = "no"`). Including it
would leak the outcome and produce a model that looks excellent but is
useless as a genuine pre-call predictor. Dropping it leaves 15 legitimate
features, still comfortably above the assignment's 12-feature minimum.

Because the target is imbalanced (~88/12), **accuracy is a misleading
headline metric** here — a trivial classifier that always predicts `no`
already scores 88.30% accuracy. AUC and MCC are given more weight in the
observations below.

## c. GitHub repository link

https://github.com/2025ac05312-eng/deposit-predictor

## d. Models used

All 5 required models were wrapped in an `sklearn.Pipeline`
(`StandardScaler` for numeric features + `OneHotEncoder(handle_unknown="ignore")`
for categoricals → classifier), tuned with 5-fold stratified `GridSearchCV`
scoring on ROC-AUC, and evaluated on a held-out, stratified 20% test split
(random seed **5312**, derived from the student ID — not the default 42).

### Comparison table — all six required metrics

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Random Forest | 0.8958 | **0.7948** | 0.6921 | 0.1975 | 0.3074 | **0.3326** |
| Logistic Regression | 0.8934 | 0.7589 | 0.6767 | 0.1701 | 0.2719 | 0.3031 |
| K-Nearest Neighbors | 0.8924 | 0.7543 | 0.6793 | 0.1522 | 0.2486 | 0.2870 |
| Naive Bayes | 0.8358 | 0.7288 | 0.3427 | **0.4395** | **0.3851** | 0.2950 |
| Decision Tree | 0.8885 | 0.7193 | 0.5541 | 0.2420 | 0.3368 | 0.3156 |

*Majority-class baseline accuracy (always predict "no"): **0.8830***

### Observations

| Model | Observation |
|---|---|
| Random Forest | Best on 3 of 6 metrics (AUC, MCC, Accuracy) and second-best Precision. Ensembling many trees reduces the variance a single Decision Tree suffers from, giving the most reliable ranking of "who is likely to subscribe." Recall is still low (0.20) — like every model here, it's conservative about calling the minority class. |
| Logistic Regression | A simple linear model lands mid-table on every metric — a strong "no-frills" baseline for a linearly-separable-ish problem, but it can't capture the non-linear interactions the tree-based models pick up (its AUC trails Random Forest by ~3.6 points). |
| K-Nearest Neighbors | Very similar profile to Logistic Regression (high precision, low recall) but the lowest MCC of all 5 models. In ~15-dimensional one-hot-expanded space, distance-based similarity is noisier than a learned decision boundary, so KNN gets the least out of this dataset. |
| Naive Bayes | The stand-out counter-example: its accuracy (0.8358) is actually **below the majority-class baseline (0.8830)** — on accuracy alone it looks like the worst model, even worse than "always guess no." But it has by far the best Recall (0.44) and F1 (0.39). Its independence assumption makes it flag far more clients as likely subscribers, trading precision for recall. This is a textbook illustration of the accuracy paradox on imbalanced data: if the bank's goal is *not missing potential subscribers*, Naive Bayes is arguably the most useful model here, despite the lowest accuracy. |
| Decision Tree | A single tree lands second-best on MCC (0.3156) and has noticeably better Recall (0.242) than Logistic Regression or KNN, but its AUC is the lowest of the five — a single tree's greedy, axis-aligned splits rank continuous probabilities worse than an ensemble or a linear model, even when its hard-label metrics look competitive. |
| **Overall Winner for your dataset?** | **Random Forest** — highest AUC and MCC, the two metrics that matter most for an imbalanced binary target, plus the best raw accuracy. *(Caveat: if the bank's priority is maximising recall — i.e., minimising missed subscribers — over precision, Naive Bayes is the stronger practical choice despite its lower accuracy.)* |

## Streamlit app link

https://deposit-predictor-yomggrttf5dyq7ptggtmgj.streamlit.app/
