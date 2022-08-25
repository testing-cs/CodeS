import sys
import os
import numpy as np

main_dir = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))
sys.path.extend([f"{main_dir}/Dataset",
                 f"{main_dir}/ModelMaker",
                 f"{main_dir}/CommonFunctions",
                 f"{main_dir}/SeqOfTokens",
                 f"{main_dir}/BagOfTokens"])

import tensorflow as tf
from ProgramArguments import *
from Utilities import *
from DsUtilities import DataRand
from SeqTokDataset import SeqTokDataset
from ModelUtils import UniqueSeed, my_ood_loss
from scipy.special import logsumexp, softmax
import pdb


def calculate_oe_score(logit_layer, x_data):
    _scores = []
    _preds = []
    _top2_diff = []

    output = logit_layer([x_data])
    smax = softmax(output, axis=1)
    _preds = np.argmax(output, axis=1)

    temp = np.sort(-1 * smax)
    top2_diff = -1 * (temp[:, 0] - temp[:, 1])
    _top2_diff.append(top2_diff)
    _scores = np.mean(output, axis=1) - logsumexp(output, axis=1)
    return _scores


def main():
    parser = makeArgParserCodeML(
        "Sequence of tokens source code classifier",
        task="classification")
    parser = addSeqTokensArgs(parser)
    parser = addRegularizationArgs(parser)
    args = parseArguments(parser)

    resetSeeds()
    DataRand.setDsSeeds(args.seed_ds)
    UniqueSeed.setSeed(args.seed_model)
    model = tf.keras.models.load_model(f"{args.model_dir}/seq-{args.data_name}-{args.metric}-oe.h5", compile=False)
    model.compile(loss=my_ood_loss, optimizer=args.optimizer, metrics=['accuracy'])
    logit_layer = tf.keras.backend.function(inputs=model.input, outputs=model.layers[-2].output)
    save_path = f"{args.result_dir}/{args.data_name}/scores"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for data_type in ['id', 'ood']:
        _ds = SeqTokDataset(f"{args.dataset}/{data_type}_test",
                            min_n_solutions=max(args.min_solutions, 3),
                            max_n_problems=None,
                            short_code_th=args.short_code,
                            long_code_th=args.long_code,
                            max_seq_length=args.seq_len,
                            test_part=1,
                            balanced_split=args.balanced_split)
        x_pad = np.zeros([len(_ds.samples), len(max(_ds.samples, key=lambda x: len(x)))])
        for i, j in enumerate(_ds.samples):
            x_pad[i][0:len(j)] = j
        split_len = len(x_pad) / 500
        scores = []
        for i in range(500):
            if i == 0:
                scores = calculate_oe_score(logit_layer, x_pad[int(i * split_len): int((i + 1) * split_len)])
            else:
                scores_s = calculate_oe_score(logit_layer, x_pad[int(i * split_len): int((i + 1) * split_len)])
                scores = np.concatenate((scores, scores_s))
        np.save(f"{save_path}/oe-{args.metric}-{data_type}.npy", scores)


if __name__ == '__main__':
    main()
