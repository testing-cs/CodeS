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
from ModelUtils import UniqueSeed


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

    save_path = f"{args.result_dir}/{args.data_name}/scores"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    model = tf.keras.models.load_model(f"{args.model_dir}/seq-{args.data_name}-{args.metric}.h5")
    for data_type in ['id', 'ood']:
        print(f"{args.metric}, {data_type}")
        if args.data_name == "java250":
            if args.metric == "problem":
                if data_type == "id":
                    problems_num = 200
                else:
                    problems_num = 50
            else:
                problems_num = 250
        elif args.data_name == "python800":
            if args.metric == "task":
                if data_type == "id":
                    problems_num = 640
                else:
                    problems_num = 160
            else:
                problems_num = 800
        else:
            if args.metric == "problem":
                if data_type == "id":
                    problems_num = 65
                else:
                    problems_num = 10
            else:
                problems_num = 75
        _ds_seq = SeqTokDataset(f"{args.dataset}/{data_type}_test",
                        min_n_solutions=max(args.min_solutions, 3),
                        max_n_problems=problems_num,
                        short_code_th=args.short_code,
                        long_code_th=args.long_code,
                        max_seq_length=args.seq_len,
                        test_part=1,
                        balanced_split=args.balanced_split)
        # test_ds_seq = _ds_seq.testDS(args.batch)
        maxlen_test = len(max(_ds_seq.samples, key=lambda x: len(x)))
        x_pad = np.zeros([len(_ds_seq.samples), maxlen_test])
        for i, j in enumerate(_ds_seq.samples):
            x_pad[i][0:len(j)] = j
        logits = model.predict(x_pad)
        scores = -np.max(logits, axis=1)
        np.save(f"{save_path}/base-{args.metric}-{data_type}.npy", scores)


if __name__ == '__main__':
    main()
