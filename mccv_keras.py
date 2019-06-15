from copy import deepcopy

from numpy import mean
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from confidence_interval import ci
from plotting import plot_history


# Monte Carlo Cross Validation
def mccv(x, y, model, plot=(), epochs=10, batch_size=10, iterations=500):
    """
    :param plot: To plot something, the first value of "plot" has to be the title of the plot, and the second has to
    be the name of the output file (without any extensions). These values are passed as a tuple.
    """
    auc = []  # ROC AUC.
    spec = []  # Proportion of actual negatives that are correctly identified as such.
    sens = []  # Proportion of actual positives that are correctly identified as such.
    history = []  # List of history objects. Used for plotting.
    for _ in tqdm(range(iterations)):  # Rewrite to "for _ in range(iterations)" to remove tqdm dependency.
        x_new, x_tst, y_new, y_tst = train_test_split(x, y, test_size=16, stratify=y)  # Test data.
        x_tr, x_val, y_tr, y_val = train_test_split(x_new, y_new, test_size=0.3, stratify=y_new)  # Train and val data.
        history.append(model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,
                                 verbose=0))
        predictions = [i[0] for i in model.predict(x=x_tst).tolist()]  # Flattening prediction list.
        auc.append(roc_auc_score(y_tst, predictions))
        report = classification_report(y_tst, [0 if p < 0.5 else 1 for p in predictions], output_dict=True)
        spec.append(report["0"]["recall"])
        sens.append(report["1"]["recall"])

    print(ci(auc))
    print(ci(spec))
    print(ci(sens))

    if plot:
        history_avg = deepcopy(history[0])
        history_avg.history["acc"] = mean([h.history["acc"] for h in history], axis=0)
        history_avg.history["loss"] = mean([h.history["loss"] for h in history], axis=0)
        history_avg.history["val_acc"] = mean([h.history["val_acc"] for h in history], axis=0)
        history_avg.history["val_loss"] = mean([h.history["val_loss"] for h in history], axis=0)
        plot_history(plot[0], history_avg, plot[1])
