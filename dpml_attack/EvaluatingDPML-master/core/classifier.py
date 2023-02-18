import os
import numpy as np
import tensorflow._api.v2.compat.v1 as tf

# from core.privacy_accountant import accountant
# from tensorflow_privacy.privacy.optimizers import dp_optimizer

LOGGING = False # enables tf.train.ProfilerHook (see use below)
LOG_DIR = 'project_log'
CHECKPOINT_DIR = '__temp_files'

AdamOptimizer = tf.train.AdamOptimizer


def get_predictions(predictions):
    """
    Returns the predicted labels and prediction scores for the inference set.
    """
    pred_y, pred_scores = [], []
    val = next(predictions, None)
    while val is not None:
        pred_y.append(val['classes'])
        pred_scores.append(val['probabilities'])
        val = next(predictions, None)
    return np.array(pred_y), np.array(pred_scores)


def get_layer_outputs(predictions):
    """
    Returns the neural network model's neuron outputs for the inference set. 
    """
    layer_outputs = []
    val = next(predictions, None)
    while val is not None:
        layer_outputs.append(val['layer_outputs'])
        val = next(predictions, None)
    return np.array(layer_outputs)


# def get_model(features, labels, mode, params):
#     """
#     Main workhorse function that defines the model according to model specification.
#     """
#     n, n_in, n_hidden, n_out, non_linearity, model, privacy, dp, epsilon, delta, batch_size, learning_rate, clipping_threshold, l2_ratio, epochs = params
#
#     if model == 'nn':
#         #print('Using neural network...')
#         input_layer = tf.reshape(features['x'], [-1, n_in])
#         h1 = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio))(input_layer)
#         h2 = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio))(h1)
#         pre_logits = tf.keras.layers.Dense(n_out, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio))(h2)
#         logits = tf.keras.layers.Softmax()(pre_logits)
#
#     elif model == 'cnn':
#         #print('Using convolution neural network...') # tailored for Cifar-100
#         input_layer = tf.reshape(features['x'], [-1, 32, 32, 3])
#         y = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation=non_linearity)(input_layer)
#         y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(y)
#         y = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation=non_linearity, input_shape=[-1, 32, 32, 3])(y)
#         y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(y)
#         y = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation=non_linearity, input_shape=[-1, 32, 32, 3])(y)
#         y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(y)
#         y = tf.keras.layers.Flatten()(y)
#         y = tf.nn.dropout(y, 0.2)
#         h1 = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio))(y)
#         h2 = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio))(h1)
#         pre_logits = tf.keras.layers.Dense(n_out, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio))(h2)
#         logits = tf.keras.layers.Softmax()(pre_logits)
#
#     else:
#         #print('Using softmax regression...')
#         input_layer = tf.reshape(features['x'], [-1, n_in])
#         logits = tf.keras.layers.Dense(n_out, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio))(input_layer)
#
#     predictions = {
#       "classes": tf.argmax(input=logits, axis=1),
#       "probabilities": logits
#     }
#     if model != 'lr':
#         predictions["layer_outputs"] = tf.concat([h1, h2, pre_logits], axis=1) # not to be used for softmax regression
#
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode,
#                                           predictions=predictions)
#
#     vector_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
#     scalar_loss = tf.reduce_mean(vector_loss)
#
#     if mode == tf.estimator.ModeKeys.TRAIN:
#
#         if privacy == 'grad_pert':
#             ac = accountant(
#                 data_size=n,
#                 batch_size=batch_size,
#                 epochs=epochs,
#                 target_delta=delta,
#                 dp_type=dp)
#             sigma = ac.get_noise_multiplier(target_epsilon=epsilon)
#             optimizer = dp_optimizer.DPAdamGaussianOptimizer(
#                             l2_norm_clip=clipping_threshold,
#                             noise_multiplier=sigma,
#                             num_microbatches=batch_size,
#                             learning_rate=learning_rate)
#             opt_loss = vector_loss
#
#         else:
#             optimizer = AdamOptimizer(learning_rate=learning_rate)
#             opt_loss = scalar_loss
#
#         global_step = tf.train.get_global_step()
#         train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
#
#         return tf.estimator.EstimatorSpec(mode=mode,
#                                           loss=scalar_loss,
#                                           train_op=train_op)
#
#     elif mode == tf.estimator.ModeKeys.EVAL:
#         eval_metric_ops = {
#             'accuracy':
#                 tf.metrics.accuracy(
#                     labels=labels,
#                      predictions=predictions["classes"])
#         }
#
#         return tf.estimator.EstimatorSpec(mode=mode,
#                                           loss=scalar_loss,
#                                           eval_metric_ops=eval_metric_ops)


# def train(dataset, n_out=None, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, clipping_threshold=1, model='nn', l2_ratio=1e-7, silent=True, non_linearity='relu', privacy='no_privacy', dp = 'dp', epsilon=0.5, delta=1e-5):
#     """
#     Calls the get_model() to create a model given the model specifications,
#     performs model training and returns the trained model (along with the
#     auxiliary information if silent != True).
#     """
#     train_x, train_y, test_x, test_y = dataset
#
#     n_in = train_x.shape[1]
#     if n_out == None:
#         n_out = len(set(np.unique(train_y)).union(set(np.unique(test_y))))
#
#     if batch_size > len(train_y):
#         batch_size = len(train_y)
#
#     if not os.path.exists(CHECKPOINT_DIR):
#         os.makedirs(CHECKPOINT_DIR)
#
#     classifier = tf.estimator.Estimator(
#             model_fn=get_model,
#             #model_dir=CHECKPOINT_DIR,
#             params = [
#                 train_x.shape[0],
#                 n_in,
#                 n_hidden,
#                 n_out,
#                 non_linearity,
#                 model,
#                 privacy,
#                 dp,
#                 epsilon,
#                 delta,
#                 batch_size,
#                 learning_rate,
#                 clipping_threshold,
#                 l2_ratio,
#                 epochs])
#
#     train_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={'x': train_x},
#         y=train_y,
#         batch_size=batch_size,
#         num_epochs=epochs,
#         shuffle=True)
#     train_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={'x': train_x},
#         y=train_y,
#         num_epochs=1,
#         shuffle=False)
#     test_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={'x': test_x},
#         y=test_y,
#         num_epochs=1,
#         shuffle=False)
#
#     steps_per_epoch = train_x.shape[0] // batch_size
#
#     if not os.path.exists(LOG_DIR):
#         os.makedirs(LOG_DIR)
#     for epoch in range(1, epochs + 1):
#         hooks = []
#         if LOGGING:
#             """
#             This hook will save traces of what tensorflow is doing
#             during the training of each model. View the combined trace
#             by running `combine_traces.py`.
#             """
#             hooks.append(tf.train.ProfilerHook(
#                 output_dir=LOG_DIR,
#                 save_steps=30))
#
#         classifier.train(input_fn=train_input_fn,
#                 steps=steps_per_epoch,
#                 hooks=hooks)
#
#         if not silent:
#             eval_results = classifier.evaluate(input_fn=train_eval_input_fn)
#             print('Train loss after %d epochs is: %.3f' % (epoch, eval_results['loss']))
#
#     if not silent:
#         eval_results = classifier.evaluate(input_fn=train_eval_input_fn)
#         train_loss = eval_results['loss']
#         train_acc = eval_results['accuracy']
#         print('Train accuracy is: %.3f' % (train_acc))
#
#         eval_results = classifier.evaluate(input_fn=test_eval_input_fn)
#         test_loss = eval_results['loss']
#         test_acc = eval_results['accuracy']
#         print('Test accuracy is: %.3f' % (test_acc))
#
#         """
#         Warning: silent flag is only used for target model training,
#         as it also returns auxiliary information.
#         """
#         return classifier, (train_loss, train_acc, test_loss, test_acc)
#
#     return classifier
