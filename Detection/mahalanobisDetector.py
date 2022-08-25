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
from lib_generation import *


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
    inter_layer = tf.keras.backend.function(inputs=model.input, outputs=model.layers[6].output)
    new_model = pureModel(model, args)
    temp_x = tf.random.normal((2, 512))
    _, temp_list = get_features(new_model, temp_x)
    feature_list = np.empty(len(temp_list))
    for out_id, out in enumerate(temp_list):
        feature_list[out_id] = out.shape[1]

    for data_type in ['id', 'ood']:
        if args.data_name == "java250":
            if args.metric == "task":
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
            if args.metric == "task":
                if data_type == "id":
                    problems_num = 65
                else:
                    problems_num = 10
            else:
                problems_num = 75
        _ds = SeqTokDataset(f"{args.dataset}/{data_type}_test",
                            min_n_solutions=max(args.min_solutions, 3),
                            max_n_problems=problems_num,
                            short_code_th=args.short_code,
                            long_code_th=args.long_code,
                            max_seq_length=args.seq_len,
                            test_part=1,
                            balanced_split=args.balanced_split)
        x_pad = np.zeros([len(_ds.samples), len(max(_ds.samples, key=lambda x: len(x)))])
        for i, j in enumerate(_ds.samples):
            x_pad[i][0:len(j)] = j
        print('get sample mean and covariance')
        sample_mean, precision = sample_estimator(new_model, inter_layer, problems_num, feature_list, x_pad, _ds.labels)
        print('get Mahalanobis scores')

        m_list = [0.0014, 0.0, 0.01, 0.005, 0.002, 0.001, 0.0005]
        for magnitude in m_list:
            layer_index = len(temp_list) - 1
            scores = get_Mahalanobis_score(new_model, inter_layer, x_pad, problems_num, sample_mean, precision, layer_index, magnitude)
            np.save(f"{save_path}/ma-{args.metric}-{data_type}-{magnitude}.npy", scores)


if __name__ == '__main__':
    main()
