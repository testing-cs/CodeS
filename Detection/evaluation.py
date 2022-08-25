import argparse
from sklearn.metrics import roc_curve, auc, average_precision_score
import numpy as np


def get_metric_scores(y_true, y_score, tpr_level):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auroc = auc(fpr, tpr)
    tpr95_pos = np.abs(tpr - tpr_level).argmin()
    tnr_at_tpr95 = 1. - fpr[tpr95_pos]
    aupr = average_precision_score(y_true, y_score)
    results = {"TNR": tnr_at_tpr95, 'AUROC': auroc, 'AUPR': aupr, "TNR_threshold": thresholds[tpr95_pos],
               'FPR': fpr, 'TPR': tpr, "threshold": thresholds}

    return results


def merge_and_generate_labels(X_pos, X_neg):

    X = np.concatenate((X_pos, X_neg))
    if len(X.shape) == 1:
        X = X.reshape((X.shape[0], -1))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='java250', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--metric', default='repo', type=str)
    parser.add_argument('--detector', default='base', type=str, choices=['base', 'oe', 'odin', 'ma'])
    args = parser.parse_args()
    detector = args.detector
    save_path = f"{args.result_dir}/{args.data_name}/scores"

    if detector in ["base", "oe"]:
        for data_type in ['id', 'ood']:
            scores = np.load(f"{save_path}/{detector}-{args.metric}-{data_type}.npy")
            if data_type == "id":
                id_scores = scores
            else:
                ood_scores = scores
        scores, labels = merge_and_generate_labels(ood_scores, id_scores)
        results = get_metric_scores(labels, scores, tpr_level=0.95)
        print(f"{args.metric}-{detector}: {results['AUROC'] * 100.}")
    elif detector == "odin":
        M_list = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
        T_list = [1, 10, 100, 1000]
        for T in T_list:
            for m in M_list:
                magnitude = m
                temperature = T
                for data_type in ['id', 'ood']:
                    scores = np.load(f"{save_path}/{detector}-{args.metric}-{data_type}-{magnitude}-{temperature}.npy")
                    if data_type == "id":
                        id_scores = scores
                    else:
                        ood_scores = scores
                scores, labels = merge_and_generate_labels(ood_scores, id_scores)
                results = get_metric_scores(labels, scores, tpr_level=0.95)
                print(f"{args.metric}-{detector}-{magnitude}-{temperature}: {results['AUROC'] * 100.}")
    else:
        m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
        for magnitude in m_list:
            for data_type in ['id', 'ood']:
                scores = np.load(f"{save_path}/{detector}-{args.metric}-{data_type}-{magnitude}.npy")
                if data_type == "id":
                    id_scores = scores
                else:
                    ood_scores = scores
            scores, labels = merge_and_generate_labels(ood_scores, id_scores)
            results = get_metric_scores(labels, scores, tpr_level=0.95)
            print(f"{args.metric}-{detector}-{magnitude}: {results['AUROC'] * 100.}")


if __name__ == '__main__':
    main()
