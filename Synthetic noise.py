import numpy as np
import matplotlib.pyplot as plt


def Noise_injector(training_label, noise_transition_matrix, class_number, plot=False):
    """
    Synthetic noise (balance dataset as default).
    :param training_label: training label of the clean dataset.
    :param noise_transition_matrix: (T) T_{ij} the probability of a sample in the i-th class
                                    flipped into the j-th class.
    :param class_number: the number of all classes.
    :param plot: print the true T and the flipped T.
    :return: the corrupted noisy labels.
    """
    noisy_label = np.zeros(shape=(training_label.shape[0],), dtype=np.int32)  # deep copy of the training labels
    for i in range(class_number):
        class_i_idx = np.where(training_label == i)[0]
        np.random.shuffle(class_i_idx)
        len_class_i_idx = class_i_idx.shape[0]
        select_idx_start = 0
        for j in range(class_number):
            T_ij = noise_transition_matrix[i, j]
            select_idx_stop = int(np.round(T_ij * len_class_i_idx)) + select_idx_start
            change_idx = class_i_idx[select_idx_start: select_idx_stop]
            noisy_label[change_idx] = j
            select_idx_start = select_idx_stop
    if plot:
        flipped_result = np.zeros(shape=(class_number, class_number))
        # the noise transition matrix of the synthetic noisy labels
        for i in range(class_number):
            class_i_idx = np.where(training_label == i)[0]
            noisy_label_i = noisy_label[class_i_idx]
            for j in range(class_number):
                noisy_label_i_j = np.where(noisy_label_i == j)[0]
                flipped_result[i, j] = noisy_label_i_j.shape[0] / class_i_idx.shape[0]

        # plot the comparison results
        label_ticks = []
        for i in range(class_number):
            label_ticks.append('{}'.format(i + 1))
        plt.figure(1)
        plt.imshow(noise_transition_matrix, cmap='viridis', interpolation='nearest')
        for i in range(len(noise_transition_matrix)):
            for j in range(len(noise_transition_matrix[i])):
                plt.text(j, i, '{:.2f}'.format(noise_transition_matrix[i][j]), ha='center', va='center', color='white')
        plt.xticks(np.arange(10, step=1), label_ticks)
        plt.yticks(np.arange(10, step=1), label_ticks)
        plt.title('true noise transition matrix')
        plt.colorbar()

        plt.figure(2)
        plt.imshow(flipped_result, cmap='viridis', interpolation='nearest')
        for i in range(len(flipped_result)):
            for j in range(len(flipped_result[i])):
                plt.text(j, i, '{:.2f}'.format(flipped_result[i][j]), ha='center', va='center', color='white')
        plt.xticks(np.arange(10, step=1), label_ticks)
        plt.yticks(np.arange(10, step=1), label_ticks)
        plt.title('flipped matrix')
        plt.colorbar()
        plt.show()

    return noisy_label


def Symmetric_noise_transition_matrix_creator(class_number, noise_rate):
    """
    Generate a noise transition matrix for symmetric noise.
    :param class_number: the number of class.
    :param noise_rate: the rate of noise labels.
    :return: symmetric noise transition matrix.
    """
    main_class = np.eye(class_number) * (1 - noise_rate)
    other_class = np.ones(shape=(class_number, class_number)) * noise_rate / class_number
    return main_class + other_class


def Asymmetric_noise_transition_matrix_creator(class_number, noise_rate):
    """
    Generate a noise transition matrix for asymmetric noise.
    :param class_number: the number of class.
    :param noise_rate: a list data for noise in the training data. for example 0->1 30%. data format[[0,1,0.3],...]
    :return: asymmetric noise transition matrix.
    """
    noise_transition_matrix = np.eye(class_number)
    for i in range(len(noise_rate)):
        ori_class = noise_rate[i][0]
        to_class = noise_rate[i][1]
        rate = noise_rate[i][2]
        noise_transition_matrix[int(ori_class), int(to_class)] = rate
        noise_transition_matrix[int(ori_class), int(ori_class)] = 1 - rate
    return noise_transition_matrix
