# [NOTE] Under development

import json

import numpy as np
import tensorflow as tf

from embeddings import Embeddings


num_features = 100
learning_rate = 0.12
batch_size = 20
num_steps = 2000

num_input =int(num_features * 3)
num_classes = 19
dropout = 0.75 # Dropout, probability to keep units


WE = Embeddings('data/glove.6B/glove.6B.100d.txt', num_features)


def fofe(sent, emb, num_features, alpha=0.5):
    x = np.zeros(num_features).astype(np.float32)
    for word in sent:
        x += alpha * emb.get(word)
    return x


def to_X(data, emb, num_features):
    X = np.zeros((len(data), int(num_features*3))).astype(np.float32)
    for i, (e1, e2, s) in enumerate(data):
        x = np.concatenate((fofe(s[e1[0]:e1[-1]+1], emb, num_features),
                            fofe(s[e2[0]:e2[-1]+1], emb, num_features),
                            fofe(s, emb, num_features)))
        X[i, :] = x
    return X


with open('result/task8_train.json', 'r', encoding='utf-8') as fd:
    train_data, train_target = json.loads(fd.read())
    train_target = [r + d if d is not None else r for r, d in train_target]
    X_train = to_X(train_data, WE, num_features)

with open('result/task8_test.json', 'r', encoding='utf-8') as fd:
    train_data, train_target = json.loads(fd.read())
    test_target = [r + d if d is not None else r for r, d in test_target]
    X_test = to_X(test_data, WE, num_features)


label2index = {l:i for i, l in enumerate(set(train_target + test_target))}
y_train = np.array([label2index[l] for l in train_target])
y_test = np.array([label2index[l] for l in test_target])


def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['features']
        x = tf.reshape(x, shape=[-1, num_input, 1])
        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv1d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling1d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv1d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling1d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'features': X_train}, y=y_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'features': X_train}, y=y_train,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)
print("Training Accuracy:", e['accuracy'])

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'features': X_test}, y=y_test,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)
print("Testing Accuracy:", e['accuracy'])
