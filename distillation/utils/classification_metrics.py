import numpy as np


def calc_f1_vectorized(precisions, recalls, eps=1e-6):

    p = np.asarray(precisions, dtype=float)
    r = np.asarray(recalls, dtype=float)

    denom = p + r
    f1 = np.zeros_like(denom, dtype=float)

    ok = (p > eps) & (r > eps) & (denom > 0)
    f1[ok] = 2.0 * p[ok] * r[ok] / denom[ok]
    return f1.tolist()


def separated_p_r_vectorized(conf_mat):

    cm = np.asarray(conf_mat, dtype=float)

    tp = np.diag(cm)              
    pred_cnt = cm.sum(axis=0)    
    true_cnt = cm.sum(axis=1)  

    p = np.divide(tp, pred_cnt, out=np.zeros_like(tp), where=(pred_cnt != 0))

    r = np.divide(tp, true_cnt, out=np.zeros_like(tp), where=(true_cnt != 0))

    return p.tolist(), r.tolist()


def metric_from_confuse_matrix(conf_mat):

    cm = np.asarray(conf_mat)
    precisions, recalls = separated_p_r_vectorized(cm)
    f1s = calc_f1_vectorized(precisions, recalls)

    logs = []
    for i, (p, r, f) in enumerate(zip(precisions, recalls, f1s)):
        segs = int(cm[i].sum())
        logs.append(
            "| label {:8d}| segs {:8d}| precision {:8.2f}| recall {:8.2f}| f1 {:8.2f}".format(
                i, segs, p, r, f
            )
        )

    logs.append(
        "| Macro| precision {:8.2f}| recall {:8.2f}| f1 {:8.2f}\n".format(
            float(np.mean(precisions)),
            float(np.mean(recalls)),
            float(np.mean(f1s)),
        )
    )

    return precisions, recalls, f1s, logs


def get_conf_mat(true_label, pred_label, num_classes=2):

    y_true = np.asarray(true_label, dtype=int)
    y_pred = np.asarray(pred_label, dtype=int)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm
