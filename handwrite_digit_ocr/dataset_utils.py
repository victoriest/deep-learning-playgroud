import numpy as np


def load_digit_dataset(*args):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for path in args:
        mnist_out = np.load(path)
        mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test = mnist_out['x_train'], \
                                                                   mnist_out['y_train'], \
                                                                   mnist_out['x_test'], \
                                                                   mnist_out['y_test']
        x_train.append(mnist_x_train)
        y_train.append(mnist_y_train)
        x_test.append(mnist_x_test)
        y_test.append(mnist_y_test)

    x_train = np.concatenate(tuple(x_train), axis=0)
    y_train = np.concatenate(tuple(y_train), axis=0)

    x_test = np.concatenate(tuple(x_test), axis=0)
    y_test = np.concatenate(tuple(y_test), axis=0)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test


def load_character_dataset(npz_dataset_path, character_range="az"):
    """
    加载英文字母数据集
    :param character_range: 字母加载范围, 只能从'a'开始
    :return:
    """
    if len(character_range) is not 2:
        raise ValueError

    if character_range[0] is not 'a':
        raise ValueError

    character_range_size = ord(character_range[1]) - ord(character_range[0]) + 1

    dataset_out = np.load(npz_dataset_path)
    x_train, y_train, x_test, y_test = dataset_out['x_train'], \
                                       dataset_out['y_train'], \
                                       dataset_out['x_test'], \
                                       dataset_out['y_test']

    if character_range_size is 26:
        return x_train, y_train, x_test, y_test

    # filter
    train_filter = np.argmax(y_train, axis=1)
    train_filter = train_filter < character_range_size
    test_filter = np.argmax(y_test, axis=1)
    test_filter = test_filter < character_range_size

    x_train_filted = x_train[train_filter]
    y_train_filted = y_train[train_filter]
    y_train_filted = np.argmax(y_train_filted, axis=1)
    y_train_filted = np.eye(character_range_size)[y_train_filted.reshape(-1)]

    x_test_filted = x_test[test_filter]
    y_test_filted = y_test[test_filter]
    y_test_filted = np.argmax(y_test_filted, axis=1)
    y_test_filted = np.eye(character_range_size)[y_test_filted.reshape(-1)]

    return x_train_filted, y_train_filted, x_test_filted, y_test_filted


if __name__ == '__main__':
    pass
    # character_range = 'az'
    # print(character_range[1], character_range[0])
    # print(ord(character_range[1]) - ord(character_range[0]))

    # (555180, 28, 28, 3) (555180, 11) (66833, 28, 28, 3) (66833, 11)
    # load_digit_dataset('D:\github.com\deep-learning-playgroud\data_pretreatment\mnist_with_space_ex.npz',
    #                    'D:\github.com\deep-learning-playgroud\data_pretreatment\mnist.npz')
