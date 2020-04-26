''' RSSI Test '''

import numpy as np
import pandas as pd

current_local = 5
next_local = 6
threshold = 30

def RSSI_test(test_data, test_label):
    fit_count = 0
    total_count = 0
    equal_list = []
    non_equal_list = []
    amount = int(test_label.shape[0])
    for i in range(amount):
        for j in range(amount - i - 1):
            if (test_label[i] == current_local):
                if (test_label[i] == test_label[i + j + 1]):
                    equal_list.append(np.linalg.norm(test_data[i] - test_data[(i + j + 1)]))
                elif (test_label[i + j + 1] == next_local):
                    non_equal_list.append(np.linalg.norm(test_data[i] - test_data[(i + j + 1)]))
                    total_count += 1
                    if (np.linalg.norm(test_data[i] - test_data[(i + j + 1)]) < threshold ):
                        fit_count += 1

    print('Equal Euclidean:', np.mean(np.array(equal_list)))
    print('Non-Equal Euclidean:', np.mean(np.array(non_equal_list)))
    print('Discard Rate:', fit_count/total_count)

def main():
    test_csv = pd.read_csv('location25_train_v1.csv')
    test = np.array(test_csv)
    test_data = -np.hsplit(test, (0, 25))[1]
    test_label = np.hsplit(test, (0, 25))[2]
    RSSI_test(test_data, test_label)

if (__name__ == '__main__'):
    main()
