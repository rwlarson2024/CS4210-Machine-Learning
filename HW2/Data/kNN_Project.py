import pandas as pd
from math import sqrt


def main():
    print("Main Function Running ...")
    correct_Predictions = 0
    data = pd.read_csv('MNIST_training.csv')
    data_ARRAY = data.to_numpy()
    # print("This is the Training Data:\n", data)
    data_Test = pd.read_csv('MNIST_test.csv')
    data_Test_ARRAY = data_Test.to_numpy()
    # print("This is the Test Data:\n", data_Test)
    K = 50

    for test_row in data_Test_ARRAY:
        prediction = KNN_prediction(data_ARRAY, test_row, K)
        if prediction == test_row[0]:
            correct_Predictions = correct_Predictions + 1
    rows = data_Test_ARRAY.shape[0]
    accuracy = (correct_Predictions / rows) * 100
    print("Estimated Accuracy: %f%%" % accuracy)


def KNN_prediction(training, testing_row, num_neighbors):
    neighbors = get_neighbors(training, testing_row, num_neighbors)
    # print (neighbors)
    output_values = [row[0] for row in neighbors]
    # print(output_values)
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def get_neighbors(training, test_row, num_neighbors):
    distances = []
    neighbors = []

    for training_row in training:
        dist = euclidean_distance(test_row[1:], training_row[1:])
        distances.append((training_row, dist))

    distances.sort(key=lambda tup: tup[1])

    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors


def euclidean_distance(row1, row2):
    distance = 0.0

    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2

    return sqrt(distance)


if __name__ == "__main__":
    main()
