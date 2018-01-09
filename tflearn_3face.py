from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import random


TYPE_SIZE = 3


def load_data(folder, folder2=None):
    """
    Loading data from folder
    :param folder: source file folder
    :param folder2: 2nd source file folder
    :return: X and y
    """
    X = []
    y = []
    idx = 0
    for i in range(TYPE_SIZE):
        filenameList = glob.glob('%s/%s/*.jpg' % (folder, i))
        y_lst = [0] * TYPE_SIZE
        y_lst[i] = 1
        for fileName in filenameList:
            X.insert(idx, np.array(Image.open(fileName)))
            y.insert(idx, np.array(y_lst))
            idx = random.randrange(len(X)+1)
        if folder2 is not None:
            filenameList2 = glob.glob('%s/%s/*.jpg' % (folder2, i))
            for fileName in filenameList2:
                X.insert(idx, np.array(Image.open(fileName)))
                y.insert(idx, np.array(y_lst))
                idx = random.randrange(len(X)+1)
    X = np.expand_dims(np.array(X) / 255.0, axis=3)
    return X, np.array(y)


def load_test(folder):
    """
    Load new data from file folder
    :param folder: folder name
    :return: X
    """
    X = []
    filenameList = glob.glob('%s/*.jpg' % folder)
    for fileName in filenameList:
        X.append(np.array(Image.open(fileName)))
    X = np.expand_dims(np.array(X) / 255.0, axis=3)
    return X


def set_cnn():
    """
    Set up the CNN architecture
    :return: CNN
    """
    # INPUT -> [[CONV3 -> RELU]*2 -> POOL2]*4 -> [FC -> RELU] -> FC
    network = input_data(shape=[None, 48, 48, 1])
    network = conv_2d(network, 32, 3, activation='relu')
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 1024, activation='relu')
    # network = dropout(network, 0.75)
    network = fully_connected(network, TYPE_SIZE, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.0005)
    return network


def pred_accuracy(Y, pY):
    """
    Calculate the accuracy of prediction
    :param Y: true value
    :param pY: predicted value
    :return: None
    """
    y_size = len(pY)
    print(y_size)
    acc_num = 0
    for i in range(y_size):
        max_val = max(pY[i])
        max_idx = pY[i].index(max_val)
        ans = Y[i].tolist().index(1)
        if max_idx == ans:
            acc_num += 1
    print('acc:', acc_num / y_size)


def plot_result(title, X_test, y_pred):
    """
    Plot the prediction result
    :param title: image title
    :param X_test: the test data
    :param y_pred: predicted reuslt
    :return: None
    """
    n = len(y_pred)
    n_0, n_1, n_2, n_3, n_4, n_5, n_6 = 0, 0, 0, 0, 0, 0, 0
    for index, (image, prediction) in enumerate(zip(X_test, y_pred)):
        prediction = prediction.index(max(prediction))
        plt.subplot(math.ceil(n/10), 10, index + 1)
        plt.axis('off')
        plt.imshow(image.reshape((48, 48)), cmap='gray', interpolation='nearest')
        # 0=Happy, 1=Sad, 2=Neutral, 3=Angry, 4=Disgust, 5=Surprise, 6=Disgust
        if prediction == 0:
            label = '0:Happy'
            n_0 += 1
        elif prediction == 1:
            label = '1:Sad'
            n_1 += 1
        elif prediction == 2:
            label = '2:Neutral'
            n_2 += 1
        elif prediction == 3:
            label = '3:Angry'
            n_3 += 1
        elif prediction == 4:
            label = '4:Disgust'
            n_4 += 1
        elif prediction == 5:
            label = '5:Surprise'
            n_5 += 1
        else:
            label = '6:Disgust'
            n_6 += 1
        plt.title('%s' % label, fontsize=6)
    plt.savefig('%s.png' % title)
    plt.clf()
    print('\n', title)
    # print(n_0, n_1, n_2, n_3, n_4, n_5, n_6)
    print(n_0, n_1, n_2)


def main():
    """
    Train CNN
    Run predictor to test data
    Plot the demo result
    :return:
    """
    X, Y = load_data('Training')
    X, X_valid, Y, Y_valid = train_test_split(X, Y, test_size=0.3)
    network = set_cnn()
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(X_valid, Y_valid),
              show_metric=True, batch_size=100,  run_id='face_cnn')

    X_test, Y_test = load_data('PublicTest')
    pY_test = model.predict(X_test)
    print("Test accuracy:")
    pred_accuracy(Y_test, pY_test)

    y_pred = []
    test_image = ['class1', 'class2', 'class3', 'class4', 'class5', 'audience1', 'audience2', 'audience3', 'audience4',
                  'concert1', 'concert2']
    for img in test_image:
        this_x = load_test(img)
        this_pred = model.predict(this_x)
        y_pred.append(this_pred)
        plot_result(img, this_x, this_pred)


if __name__ == '__main__':
    main()
