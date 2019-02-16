import matplotlib.pyplot as plt
import pylab
import numpy as np


class Network:
    def __init__(self, fuzzy_file_name, weigths_file_name1, weights_file_name2, learning_speed, number_of_batches):
        self.fuzzy_table, self.weigths_table1, self.weigths_table2 = read_tables(fuzzy_file_name=fuzzy_file_name,
                                                                                 weights_file_name1=weigths_file_name1,
                                                                                 weights_file_name2=weights_file_name2)
        self.learning_speed = learning_speed
        self.av_gradients1 = [[0 for x in range(3)] for y in range(2)]
        self.av_gradients2 = [[0 for x in range(3)] for y in range(1)]
        self.number_of_batches = number_of_batches

    def learning(self):
        residual = len(self.fuzzy_table) % self.number_of_batches
        errors = []
        training_samples_size = len(self.fuzzy_table) - residual
        step = 1
        av_errors = []
        av_error = 0
        for i in range(training_samples_size):
            z1, z2, predicted, y = self.forward(position=i)
            error = (y - predicted) ** 2
            av_error += error
            errors.append(error)
            self.backward(position=i, z1=z1, z2=z2, predicted=predicted, y=y)
            if (step % self.number_of_batches == 0):
                self.batch_round()
                av_error /= self.number_of_batches
                av_errors.append(av_error)
                av_error=0
            step=step+1
        plt.plot(av_errors)
        plt.show()

    def forward(self, position):
        z1 = 0
        z2 = 0
        y = 0
        z1 = self.fuzzy_table[position][0] * self.weigths_table1[0][0] + self.fuzzy_table[position][1] * \
             self.weigths_table1[1][
                 0] + \
             self.weigths_table1[0][2]
        if z1 < 0:
            z1 = 0
        if z2 < 0:
            z2 = 0
        z2 = self.fuzzy_table[position][0] * self.weigths_table1[0][1] + self.fuzzy_table[position][1] * \
             self.weigths_table1[1][
                 1] + \
             self.weigths_table1[1][2]
        predicted = z1 * self.weigths_table2[0][0] + z2 * self.weigths_table2[0][1] + self.weigths_table2[0][2]
        y = self.fuzzy_table[position][2]
        return z1, z2, predicted, y

    def backward(self, position, z1, z2, predicted, y):
        l_grad = 2 * (predicted - y)
        z1_grad = l_grad * self.weigths_table2[0][0]
        z2_grad = l_grad * self.weigths_table2[0][1]
        # gradients for lambda
        """
        self.weigths_table2[0][0] = self.weigths_table2[0][0] - self.learning_speed * l_grad * z1
        self.weigths_table2[0][1] = self.weigths_table2[0][1] - self.learning_speed * l_grad * z2
        self.weigths_table2[0][2] = self.weigths_table2[0][2] - self.learning_speed * l_grad * 1
        # gradients for z1
        self.weigths_table1[0][0] = self.weigths_table1[0][0] - self.learning_speed * z1_grad * \
                                    self.fuzzy_table[position][0]
        self.weigths_table1[0][1] = self.weigths_table1[0][1] - self.learning_speed * z1_grad * \
                                    self.fuzzy_table[position][1]
        self.weigths_table1[0][2] = self.weigths_table1[0][2] - self.learning_speed * z1_grad * \
                                    self.fuzzy_table[position][2]
        # gradients for z2
        self.weigths_table1[1][0] = self.weigths_table1[1][0] - self.learning_speed * z2_grad * \
                                    self.fuzzy_table[position][0]
        self.weigths_table1[1][1] = self.weigths_table1[1][1] - self.learning_speed * z2_grad * \
                                    self.fuzzy_table[position][1]
        self.weigths_table1[1][2] = self.weigths_table1[1][2] - self.learning_speed * z2_grad * \
                                    self.fuzzy_table[position][2]
        """
        self.av_gradients2[0][0] = self.av_gradients2[0][0] + (
                l_grad * z1)
        self.av_gradients2[0][1] = self.av_gradients2[0][1] + (
                l_grad * z2)
        self.av_gradients2[0][2] = self.av_gradients2[0][2] + (
                l_grad * 1)
        # gradients for z1
        self.av_gradients1[0][0] = self.av_gradients1[0][0] + (
                z1_grad * \
                self.fuzzy_table[position][0])
        self.av_gradients1[0][1] = self.av_gradients1[0][1] + (
                z1_grad * \
                self.fuzzy_table[position][1])
        self.av_gradients1[0][2] = self.av_gradients1[0][2] + (
                z1_grad * \
                self.fuzzy_table[position][2])
        # gradients for z2
        self.av_gradients1[1][0] = self.av_gradients1[1][0] + (
                z2_grad * \
                self.fuzzy_table[position][0])
        self.av_gradients1[1][1] = self.av_gradients1[1][1] + (
                z2_grad * \
                self.fuzzy_table[position][1])
        self.av_gradients1[1][2] = self.av_gradients1[1][2] + (
                z2_grad * \
                self.fuzzy_table[position][2])

    def batch_round(self):
        for i in range(len(self.av_gradients1)):
            for j in range(len(self.av_gradients1[i])):
                self.weigths_table1[i][j] = self.weigths_table1[i][j] - (self.av_gradients1[i][j] / self.number_of_batches)*self.learning_speed
                self.av_gradients1[i][j] = 0
        for i in range(len(self.av_gradients2)):
            for j in range(len(self.av_gradients2[i])):
                self.weigths_table2[i][j] = self.weigths_table2[i][j] - (self.av_gradients2[i][j] / self.number_of_batches)*self.learning_speed
                self.av_gradients2[i][j] = 0


def read_tables(fuzzy_file_name, weights_file_name1, weights_file_name2):
    num_lines = sum(1 for line in open(fuzzy_file_name))

    weigths_table1 = [[0 for x in range(3)] for y in range(2)]
    weigths_table2 = [[0 for x in range(3)] for y in range(1)]
    fuzzy_table = [[0 for x in range(3)] for y in range(num_lines)]
    fill_array(filename=fuzzy_file_name, array=fuzzy_table)
    fill_array(filename=weights_file_name1, array=weigths_table1)
    fill_array(filename=weights_file_name2, array=weigths_table2)
    return fuzzy_table, weigths_table1, weigths_table2


def fill_array(filename, array):
    f = open(filename)
    i = 0
    for line in f:
        j = 0
        for number in line.split():
            array[i][j] = float(number)
            j = j + 1
        i = i + 1
    f.close()


def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print('[' + str(matrix[i][j]) + "] ", end='')
        print()
    print("")


"""network = Network("fuzzy.txt", "weigths1.txt", "weigths2.txt", 0.001, 100)
print("weights1:")
print_matrix(network.weigths_table1)
print("weights2:")
print_matrix(network.weigths_table2)
network.learning()
print("weights1 after learning:")
print_matrix(network.weigths_table1)
print("weights2 after learning:")
print_matrix(network.weigths_table2)"""

cnums = np.arange(5) + 1j * np.arange(6,11)
X = [x.real for x in cnums]
Y = [x.imag for x in cnums]
plt.scatter(X,Y, color='red')
plt.show()