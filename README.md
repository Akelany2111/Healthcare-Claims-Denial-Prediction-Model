# Healthcare-Claims-Denial-Prediction-Model
Project overview

This project uses synthetic healthcare claims data to predict whether a claim is likely to be denied. The goal is to identify denial risk before claim adjudication so that billing or operations teams can proactively review high-risk claims and reduce revenue leakage.

Business problem

Denied claims create administrative burden, delay reimbursement, and reduce financial performance. A predictive model can help flag higher-risk claims before submission or during early review so teams can prioritize interventions.

Data sources

The project uses three synthetic datasets:

claims.csv
providers.csv
insurance.csv

These were merged into a single modeling dataset with claim-level, provider-level, and payer-level attributes.

Features used

Pre-submission features included:

provider name
specialty
clinic location
payer name
payer type
claim profile
billed amount
service month / day-of-week
days between service and submission
Important modeling note

Initial model testing revealed target leakage from post-outcome fields such as claim_status and paid_amount. These fields were removed from the final model to ensure the model only used information available before denial outcome.

Models tested
Logistic Regression
Random Forest Classifier
Evaluation metrics
Accuracy
Precision
Recall
F1-score
ROC AUC
Key learning

The project demonstrated the importance of:

feature engineering
class imbalance handling
leakage detection
interpreting model outputs in a business context
Tools used
Python
pandas
scikit-learn
matplotlib
Business value

This model can support:

denial risk monitoring
claims prioritization
operational workflow improvement
revenue protection

IMPOTANT NOTE " After removing target leakage, the model’s feature importance became much more operationally meaningful. Variables such as billed amount, days to submit, service timing, claim profile, and payer characteristics emerged as the primary drivers of denial risk. However, the current model was not yet effective at correctly classifying denied claims, as shown by the confusion matrix and ROC AUC. This suggests that either the synthetic dataset does not yet contain strong enough denial patterns, or that further feature engineering and class imbalance handling are needed to make the model operationally useful."
