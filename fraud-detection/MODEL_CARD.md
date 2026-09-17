# Model Card

## Intended use

Rank card transactions for a constrained review queue and estimate scenario-level operational value. The project is a reproducible research demonstration, not a production authorization service.

## Candidate models

- unweighted, class-weighted and SMOTE logistic regression
- cost-sensitive XGBoost and LightGBM
- Balanced Random Forest
- EasyEnsemble
- PyTorch multilayer perceptron with weighted BCE and focal-loss challengers

Model families are tuned with expanding temporal folds. Family selection uses a later chronological validation block. Calibration, policy selection and final evaluation use separate later blocks.

## Evaluation

Primary ranking metric: PR-AUC. Secondary metrics include ROC-AUC, PR-AUC lift over prevalence, Brier score, Brier skill, log loss, MCC, precision, recall and business cost. The final test block is evaluated in one pipeline stage after model, calibration and policy choices are frozen.

## Known risks

- extreme rarity makes estimates uncertain;
- the short observation window limits drift conclusions;
- PCA features are not business-interpretable;
- cost parameters are assumptions;
- manual-review capacity is modeled in elapsed-hour batches;
- calibration can drift when the fraud base rate changes.

Production use would require recent institution-specific data, delayed-label handling, monitoring, challenger governance, fairness review, security controls and controlled rollout.

