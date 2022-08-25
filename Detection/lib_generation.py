import sklearn.covariance
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import pdb


def pureModel(model, args):
    input_shape = (512)
    input_tensor = layers.Input(shape=input_shape)
    x = layers.Dense.from_config(model.layers[7].get_config())(input_tensor)
    x = layers.Dense.from_config(model.layers[8].get_config())(x)
    x = layers.Dense.from_config(model.layers[9].get_config())(x)
    x = layers.Dense.from_config(model.layers[10].get_config())(x)
    new_model = Model(input_tensor, x)
    new_model.compile(loss="sparse_categorical_crossentropy", optimizer=args.optimizer, metrics=['accuracy'])
    return new_model


def get_features(model, input_x):
    out_list = []
    x = layers.Dense.from_config(model.layers[1].get_config())(input_x)
    out_list.append(x)
    x = layers.Dense.from_config(model.layers[2].get_config())(x)
    out_list.append(x)
    x = layers.Dense.from_config(model.layers[3].get_config())(x)
    out_list.append(x)
    x = layers.Dense.from_config(model.layers[4].get_config())(x)
    out_list.append(x)

    return x, out_list


def sample_estimator(model, inter_layer, num_classes, feature_list, x_pad, labels):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
            precision: list of precisions
    """
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    split_len = len(x_pad) / 50

    for data_split in range(50):
        x_data = tf.convert_to_tensor(x_pad[int(data_split * split_len): int((data_split + 1) * split_len)])
        new_input = tf.convert_to_tensor(inter_layer(x_data))
        target = labels[int(data_split * split_len): int((data_split + 1) * split_len)]
        total += len(target)
        output, out_features = get_features(model, new_input)

        # get hidden features
        for i in range(len(feature_list)):
            out_features[i] = tf.reshape(out_features[i], (out_features[i].shape[0], out_features[i].shape[1], -1))
            out_features[i] = np.mean(out_features[i], 2)

        # compute the accuracy
        pred = np.argmax(output, axis=1)
        correct += np.sum(pred == target)

        # construct the sample matrix
        for i in range(new_input.shape[0]):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = tf.reshape(out[i], (1, -1))
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = np.concatenate((list_features[out_count][label], tf.reshape(out[i], (1, -1))), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = np.zeros((num_classes, int(num_feature)))
        for j in range(num_classes):
            temp_list[j] = np.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = np.concatenate((X, list_features[k][i] - sample_class_mean[k][i]), axis=0)

        # find inverse
        group_lasso.fit(X)
        temp_precision = group_lasso.precision_
        precision.append(temp_precision)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


def get_Mahalanobis_score(model, inter_layer, x_pad, num_classes, sample_mean, precision, layer_index, magnitude):
    """
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    """
    Mahalanobis = []
    split_len = len(x_pad) / 50
    for data_split in range(50):
        x_data = tf.convert_to_tensor(x_pad[int(data_split * split_len): int((data_split + 1) * split_len)])
        new_input = tf.convert_to_tensor(inter_layer(x_data))

        with tf.GradientTape() as tape:
            tape.watch(new_input)
            output, out_features_all = get_features(model, new_input)
            out_features = out_features_all[layer_index]
            out_features = tf.reshape(out_features, (out_features.shape[0], out_features.shape[1], -1))
            out_features = tf.math.reduce_mean(out_features, 2)

            # compute Mahalanobis score
            gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = out_features - batch_sample_mean
                term_gau = -0.5 * np.diag(tf.linalg.matmul(tf.linalg.matmul(zero_f, precision[layer_index]), tf.transpose(zero_f)))
                if i == 0:
                    gaussian_score = tf.reshape(term_gau, (-1, 1))
                else:
                    gaussian_score = tf.concat((gaussian_score, tf.reshape(term_gau, (-1, 1))), axis=1)

            # Input_processing
            sample_pred = tf.math.argmax(gaussian_score, axis=1)
            batch_sample_mean_ = tf.experimental.numpy.take(sample_mean[layer_index], sample_pred, axis=0)
            batch_sample_mean = tf.cast(batch_sample_mean_, tf.float32)
            zero_f = out_features - batch_sample_mean
            pure_gau = -0.5 * tf.linalg.diag(tf.linalg.matmul(tf.linalg.matmul(zero_f, precision[layer_index]), tf.transpose(zero_f)))
            loss = tf.math.reduce_mean(-pure_gau)
        grads = tape.gradient(loss, new_input)
        grads = (grads - 0.5) * 2
        x_data_pert = new_input - magnitude * grads
        _, noise_out_features_all = get_features(model, x_data_pert)
        noise_out_features = noise_out_features_all[layer_index]
        noise_out_features = tf.reshape(noise_out_features, (noise_out_features.shape[0], noise_out_features.shape[1], -1))
        noise_out_features = tf.math.reduce_mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features - batch_sample_mean
            term_gau = -0.5 * np.diag(tf.linalg.matmul(tf.linalg.matmul(zero_f, precision[layer_index]), tf.transpose(zero_f)))
            if i == 0:
                noise_gaussian_score = tf.reshape(term_gau, (-1, 1))
            else:
                noise_gaussian_score = tf.concat((noise_gaussian_score, tf.reshape(term_gau, (-1, 1))), 1)
        noise_gaussian_score = -np.max(noise_gaussian_score, axis=1)
        Mahalanobis.extend(noise_gaussian_score)
    return Mahalanobis

