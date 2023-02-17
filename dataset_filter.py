import pandas as pd
import numpy as np
import os


'''choose data by filter file'''


def dataset_filter(dataset_name='location30',sheet_name = '0'):
    save_path = './ml_privacy_meter_result/' + dataset_name
    fpath = "./ratio_filter/" + dataset_name + "/ratio_filter.xlsx"

    df = pd.read_excel(fpath, sheet_name=sheet_name)
    index_list = df.iloc[:,0]
    #index_list = df['index']
    # print(type(index_list))
    index_list = index_list.values
    index_list = np.array(list(filter(lambda x: not np.isnan(x), [x for x in index_list])))  # 去除nan
    index_list = index_list.astype(int)
    index_list = index_list.tolist()

    # print(index_list)
    # print(type(index_list))
    new_index_list = [j for i in index_list for j in range((i - 1) * 10, i*10)]

    if dataset_name == 'purchase100':
        shadow_index_list = [j for i in index_list for j in range(int((i - 1) * 10 + 197324*0.05), int(i * 10 + 197324*0.05))]
        new_test_index_list = [j for i in index_list for j in range(int((i - 1) * 10+197324*0.4), int(i*10+197324*0.4))]
    elif dataset_name == 'texas100':
        shadow_index_list = [j for i in index_list for j in range(int((i - 1) * 10 + 5000),int(i * 10 + 5000))]
        new_test_index_list = [j for i in index_list for j in range(int((i - 1) * 10 +  67330*0.3+10000), int(i * 10 +  67330*0.3+10000))]

    '''structure data'''

    # X_target = X[new_index_list]
    # Y_target = Y[new_index_list]
    #

    # X_shadow = X[shadow_index_list]
    # Y_shadow = Y[shadow_index_list]
    #

    # X_test = X[new_test_index_list]
    # Y_test = Y[new_test_index_list]
    # print()


    '''image data'''

    X_target = x_train[new_index_list]
    Y_target = y_train[new_index_list]

    #shadow model
    X_shadow = x_train[new_index_list]
    Y_shadow = y_train[new_index_list]

    X_test = x_test[new_index_list]
    Y_test = y_test[new_index_list]

    import tensorflow as tf

    import sys
    from privacy_meter.audit import Audit, MetricEnum
    from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
    from privacy_meter.constants import InferenceGame
    from privacy_meter.dataset import Dataset
    from privacy_meter.information_source import InformationSource
    from privacy_meter.model import TensorflowModel

    seed = 1234
    np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)

    # for training the target and reference models
    num_points_per_train_split = 5000
    num_points_per_test_split = 1000
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optim_fn = 'adam'
    epochs = 10
    batch_size = 64
    regularizer_penalty = 0.01
    regularizer = tf.keras.regularizers.l2(l=regularizer_penalty)

    # for the reference metric
    num_reference_models = 10
    fpr_tolerance_list = [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ]

    def preprocess_cifar100_dataset():
        #input_shape, num_classes = (32, 32, 3), 100
         
        # split the data between train and test sets
        # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x_train = X_target
        y_train = Y_target

        x_test = X_test
        y_test = Y_test
        # scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        # convert labels into one hot vectors
        # y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        # y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        input_shape, num_classes = x_train[0].shape, len(y_train[0])
        
        return x_train, y_train, x_test, y_test, input_shape, num_classes

    x_train_all, y_train_all, x_test_all, y_test_all, input_shape, num_classes = preprocess_cifar100_dataset()

    print(x_train_all.shape, x_test_all.shape)

    # create the target model's dataset
    dataset = Dataset(
        data_dict={
            'train': {'x': x_train_all, 'y': y_train_all},
            'test': {'x': x_test_all, 'y': y_test_all}
        },
        default_input='x',
        default_output='y'
    )

    datasets_list = dataset.subdivide(
        num_splits=num_reference_models + 1,
        delete_original=True,
        in_place=False,
        return_results=True,
        method='hybrid',
        split_size={'train': num_points_per_train_split, 'test': num_points_per_test_split}
    )

    for i, d in enumerate(datasets_list):
        print(i)
        print(d)

    def get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer):
        # TODO: change model architecture
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                         input_shape=input_shape, kernel_regularizer=regularizer))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                                         kernel_regularizer=regularizer))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        return model

    x = datasets_list[0].get_feature('train', '<default_input>')
    y = datasets_list[0].get_feature('train', '<default_output>')
    model = get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer)
    model.summary()
    model.compile(optimizer=optim_fn, loss=loss_fn, metrics=['accuracy'])
    model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=2)

    target_model = TensorflowModel(model_obj=model, loss_fn=loss_fn)

    reference_models = []
    for model_idx in range(num_reference_models):
        print(f"Training reference model {model_idx}...")
        reference_model = get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer)
        reference_model.compile(optimizer=optim_fn, loss=loss_fn, metrics=['accuracy'])
        reference_model.fit(
            datasets_list[model_idx + 1].get_feature('train', '<default_input>'),
            datasets_list[model_idx + 1].get_feature('train', '<default_output>'),
            batch_size=batch_size,
            epochs=epochs,
            verbose=2
        )
        reference_models.append(
            TensorflowModel(model_obj=reference_model, loss_fn=loss_fn)
        )

    target_info_source = InformationSource(
        models=[target_model],
        datasets=[datasets_list[0]]
    )

    reference_info_source = InformationSource(
        models=reference_models,
        datasets=[datasets_list[0]]  # we use the same dataset for the reference models
    )

    audit_obj = Audit(
        metrics=MetricEnum.REFERENCE,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        fpr_tolerances=fpr_tolerance_list,
        save_path:str = save_path
    )
    audit_obj.prepare()

    audit_results = audit_obj.run()[0]
    for result in audit_results:
        print(result)


    '''add attack in there'''


#dataset_names = ['purchase100','texas100','CH_MNIST','CIFAR100','CIFAR10','imagenet']
#dataset_names = ['texas100']
dataset_names = ['CH_MNIST','CIFAR100','CIFAR10','imagenet']
#dataset_names = ['CIFAR100']
for dataset_name in dataset_names:
    if dataset_name == 'location30':
        path = '.\datasets\location\shuffle_index.npz'
        shuffle_index_list = np.load(path, allow_pickle=True)['x']
        path = '.\datasets\location\data_complete.npz'
        dataset = np.load(path, allow_pickle=True)
        X = dataset['x'][shuffle_index_list]
        Y = dataset['y'][shuffle_index_list]
    elif(dataset_name == 'purchase100'):
        path = './dataset_shuffle/random_r_purchase100.npy'
        shuffle_index_list = np.load(path, allow_pickle=True)
        path = './datasets/purchase/dataset_purchase'
        data_set = np.genfromtxt(path, delimiter=',')
        X = data_set[:, 1:].astype(np.float64)
        Y = (data_set[:, 0]).astype(np.int32) - 1
        X = X[shuffle_index_list]
        Y = Y[shuffle_index_list]

    elif(dataset_name == 'texas100'):
        path = './dataset_shuffle/random_r_texas100.npy'
        shuffle_index_list = np.load(path, allow_pickle=True)
        DATASET_PATH = './datasets/texas/'
        DATASET_FEATURES = os.path.join(DATASET_PATH, 'texas/100/feats')
        DATASET_LABELS = os.path.join(DATASET_PATH, 'texas/100/labels')
        data_set_features = np.genfromtxt(DATASET_FEATURES, delimiter=',')
        data_set_label = np.genfromtxt(DATASET_LABELS, delimiter=',')

        X = data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32) - 1
        X = X[shuffle_index_list]
        Y = Y[shuffle_index_list]


    else:
        data_path = './image_datasets/' + dataset_name + '/Target_data/' + dataset_name
        # (x_train, y_train), (x_test, y_test) =
        x_train = np.load(data_path + '_x_train.npy')
        # print(f'{x_train[0].shape=}')
        y_train = np.load(data_path + '_y_train.npy')
        x_test = np.load(data_path + '_x_test.npy')
        y_test = np.load(data_path + '_y_test.npy')

    print('{dataset_name=}' + dataset_name)
    fpath = "./ratio_filter/" + dataset_name + "/ratio_filter.xlsx"
    reader = pd.ExcelFile(fpath)
    sheet_names = reader.sheet_names
    for sheet_name in sheet_names:
        save_path = './ml_privacy_meter_result/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(save_path + dataset_name, 'a') as f:
            print('sheet_name= ' + sheet_name, file=f)
        dataset_filter(dataset_name=dataset_name, sheet_name=sheet_name)
        # try:
        #     dataset_filter(dataset_name=dataset_name,sheet_name = sheet_name)
        # except:
        #     print('error in '+sheet_name)