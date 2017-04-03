import argparse
import numpy
import csv

output = ''
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 0.5]
iters = [100, 100, 100, 100, 100, 100, 100, 100, 100, 1000]


def write_to_csv(rate, iterations, b_0, b_age, b_weight):

    with open(output, 'a') as my_file:
        writer = csv.writer(my_file, lineterminator='\n')
        writer.writerow([rate, iterations, b_0, b_age, b_weight])

    my_file.close()


def prepare(data):

    features = data[:, :2]
    labels = data[:, 2]
    normalized = features

    # Scale each feature by its standard deviation and set its mean to zero
    for i in range(features.shape[1]):

        mean = numpy.mean(features[:, i])

        st_dev = numpy.std(features[:, i])

        normalized[:, i] = (normalized[:, i] - mean) / st_dev

    # Add the intercept in front of the features matrix (vector 1)
    samples = len(labels)

    features = numpy.ones(shape=(samples, 3))

    features[:, 1:3] = normalized

    return features, labels


def cost(features, labels, beta):
    return numpy.sum((features.dot(beta) - labels)**2) / 2 / len(labels)


def descent(features, labels, beta, rate, iterations):

    history = [0] * iterations

    samples = len(labels)

    for iteration in range(iterations):

        prediction = features.dot(beta)

        loss = prediction - labels

        gradient = features.T.dot(loss) / samples

        beta = beta - rate * gradient

        history[iteration] = cost(features, labels, beta)

    return beta, history


def main():
    global output, learning_rates, iters

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    data = numpy.genfromtxt(args.input, delimiter=',')

    output = args.output

    features, labels = prepare(data)

    betas = numpy.array([0, 0, 0])

    for iteration, rate in zip(iters, learning_rates):

        beta, history = descent(features, labels, betas, rate, iteration)

        write_to_csv(rate, iteration, beta[0], beta[1], beta[2])

        # print([beta, history])

if __name__ == '__main__':
    main()
