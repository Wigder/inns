from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

from confidence_interval import ci
from load_data import x, y

# Monte Carlo Cross Validation
iterations = 500
auc = []  # ROC AUC.
spec = []  # Proportion of actual negatives that are correctly identified as such.
sens = []  # Proportion of actual positives that are correctly identified as such.
for _ in tqdm(range(iterations)):  # Rewrite to "for _ in range(iterations)" to remove tqdm dependency.
    x_tr, x_tst, y_tr, y_tst = train_test_split(x, y, test_size=(1 - 0.698), stratify=y)  # Stratified split.
    model = SVC(gamma="scale", probability=True)
    # model = LogisticRegression(solver="liblinear")  # Uncomment to use logistic regression instead.
    model.fit(x_tr, y_tr)
    auc.append(roc_auc_score(y_tst, model.predict_proba(x_tst)[:, 1]))
    report = classification_report(y_tst, model.predict(x_tst), output_dict=True)
    spec.append(report["0"]["recall"])
    sens.append(report["1"]["recall"])

print(ci(auc))
print(ci(spec))
print(ci(sens))
