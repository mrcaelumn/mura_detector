import tensorflow as tf
from sklearn.metrics import roc_curve
import pandas as pd


def get_tnr_tpr_custom(labels, scores, tnr_min=0.9):
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # get the best threshold with fpr <=0.1
    df = pd.DataFrame(columns=['threshold', 'TNR', 'TPR'])

    for th in thresholds:

        m_scores_ano_final = (scores > th).astype(float)

        m_test_cm = tf.math.confusion_matrix(
            labels=labels,
            predictions=m_scores_ano_final
        ).numpy()

        M_T_TP = m_test_cm[1][1]
        M_T_FP = m_test_cm[0][1]
        M_T_FN = m_test_cm[1][0]
        M_T_TN = m_test_cm[0][0]

        TNR = (M_T_TN / (M_T_FP + M_T_TN))
        TPR = (M_T_TP / (M_T_TP + M_T_FN))

        # print(data)
        if TNR >= tnr_min:
            data = {
                "threshold": float(th),
                "TNR": float(TNR),
                "TPR": float(TPR),
            }
            df = pd.concat([df, pd.DataFrame.from_records([data])])

    test = df.sort_values('TPR', ascending=False)
    print(test.head(3))
    best_threshold = test['threshold'].iloc[0]
    best_TNR = test['TNR'].iloc[0]
    best_TPR = test['TPR'].iloc[0]

    return best_threshold, best_TNR, best_TPR
