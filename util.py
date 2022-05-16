
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#from project 1. Used for plotting scores.
def plotter(y_value, x_title, y_title, title, legend, filename):
    '''
    function for producing graphs
    '''

    fig, ax = plt.subplots()

    line_labels = legend

    max_len = 0

    # plotting multiple lines
    for idx in range(len(y_value)):
        x_value = np.arange(len(y_value[idx]))

        if max_len < len(y_value[idx]):
            max_len = len(y_value[idx])
        ax.plot(x_value, y_value[idx])

    # set the title and labels
    ax.set(xlabel=x_title, ylabel=y_title, title=title)
    plt.legend(line_labels, loc='lower right', fancybox=True, shadow=True)

    fig.savefig(filename)


def train_valid_test_split(file_names, seeds=666):

    train_list = []
    valid_list = []
    test_list = []

    for dirs in file_names:
        # print(len(dirs))

        temp_ind_array = np.arange(len(dirs))
        X_train, X_test, _, _ = train_test_split(temp_ind_array, temp_ind_array, test_size=0.3, random_state=seeds)
        X_valid, X_test, _, _ = train_test_split(X_test, X_test, test_size=0.5, random_state=seeds)

        temp_file_arrays = np.asarray(dirs)

        train_list.append(temp_file_arrays[[X_train]])
        valid_list.append(temp_file_arrays[[X_valid]])
        test_list.append(temp_file_arrays[[X_test]])

    return train_list, valid_list, test_list


def combine_lists(file_list):
    '''
    Converts list of list into a list
    :param file_list:
    :return:
    '''

    temp_list = []

    for f in file_list:
        for name in f:
            temp_list.append(name)

    return temp_list