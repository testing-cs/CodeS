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
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


def pureModel(model, args):
    input_shape = (512)
    input_tensor = layers.Input(shape=input_shape)
    x = layers.Dropout.from_config(model.layers[6].get_config())(input_tensor)
    x = layers.Dense.from_config(model.layers[7].get_config())(x)
    x = layers.Dense.from_config(model.layers[8].get_config())(x)
    x = layers.Dense.from_config(model.layers[9].get_config())(x)
    x = layers.Dense.from_config(model.layers[10].get_config())(x)
    new_model = Model(input_tensor, x)
    new_model.compile(loss="sparse_categorical_crossentropy", optimizer=args.optimizer, metrics=['accuracy'])
    return new_model


def calculate_odin_score(model, x_data, magnitude=0.0014, temperature=1000):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(x_data)
        preds = model(x_data) / temperature
        loss = loss_object(tf.reduce_max(preds, axis=1), preds)
    grads = tape.gradient(loss, x_data)
    grads = (grads - 0.5) * 2
    # add small perturbations to data
    x_data_pert = x_data - magnitude * grads
    outputs = model.predict(x_data_pert) / temperature
    nnOutputs = softmax(outputs)
    scores = -np.max(nnOutputs, axis=1)
    return scores


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
            if args.metric == "problem":
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
        model = tf.keras.models.load_model(f"{args.model_dir}/seq-{args.data_name}-{args.metric}.h5", compile=False)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=args.optimizer, metrics=['accuracy'])
        inter_layer = tf.keras.backend.function(inputs=model.input, outputs=model.layers[5].output)
        new_model = pureModel(model, args)
        split_len = len(x_pad) / 50
        scores = []
        M_list = [0.0014, 0.0005, 0.001, 0, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
        T_list = [1000, 10, 100, 1]
        for T in T_list:
            for m in M_list:
                magnitude = m
                temperature = T
                for i in range(50):
                    x_data = tf.convert_to_tensor(x_pad[int(i * split_len): int((i + 1) * split_len)])
                    new_input = tf.convert_to_tensor(inter_layer(x_data))
                    if i == 0:
                        scores = calculate_odin_score(new_model, new_input, magnitude=magnitude, temperature=temperature)
                    else:
                        scores_s = calculate_odin_score(new_model, new_input, magnitude=magnitude, temperature=temperature)
                        scores = np.concatenate((scores, scores_s))
                    np.save(f"{save_path}/odin-{args.metric}-{data_type}-{magnitude}-{temperature}.npy", scores)


if __name__ == '__main__':
    main()
