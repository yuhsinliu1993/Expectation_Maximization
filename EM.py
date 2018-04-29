import dataset
import argparse
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from data_generator import GaussianNoise

eps = np.finfo(float).eps

def bernoulli_likelihood(mu, x):
    l = 1
    for i in range(len(x)):
        l *= (mu[i] ** x[i]) * ((1 - mu[i]) ** (1 - x[i]))
    return l


def clustering(predictions, labels):
    clusters = np.zeros((10, 10), dtype=np.int)
    for index, item in enumerate(predictions):
        clusters[int(labels[index]), int(item)] += 1
    return clusters

def plot_means(means):

    k = means.shape[0]

    rows = k // 5 + 1
    columns = min(k, 5)

    for i in range(k):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(scipy.misc.toimage(means[i].reshape(28, 28), cmin=0.0, cmax=1.0).convert('RGBA'))
    plt.show()


class Bernoulli_EM:

    def __init__(self, num_classes, num_iteration=100, tol=1e-3, verbose=False):

        self.num_classes = num_classes
        self.num_iteration = num_iteration

        # convergence threshold
        self.tol = tol

        self.verbose = verbose

        self.pi = np.array([1 / num_classes for _ in range(num_classes)])
        self.mu = None

        self._converged = False

    def fit(self, x):
        print("[+] Trainging ...")
        dim = x.shape[1]

        self.mu = np.ndarray(shape=(self.num_classes, dim))
        self.mu = np.random.rand(self.num_classes, dim) * 0.5 + 0.25

        iterations = 0

        prev_log_likelihood = None
        current_log_likelihood = -np.inf

        while iterations < self.num_iteration:
            prev_log_likelihood = current_log_likelihood

            # E_step
            log_likelihoods, Qz = self.score_samples(x)
            current_log_likelihood = log_likelihoods.mean()

            if self.verbose:
                print('[{:02d}] likelihood = {}'.format(iterations, current_log_likelihood))

            if prev_log_likelihood is not None:
                if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                    self._converged = True
                    break

            self.M_step(x, Qz)

            iterations += 1

        if self._converged:
            print('converged in {} iterations'.format(iterations))

    def M_step(self, x, Qz):
        """
        Args:
        -----
        x:  data
        Qz: the distribution over latent variables z  shape = (N, 10)
        """
        weights_sum = Qz.sum(axis=0)       # (10, )
        weighted_x_sum = np.dot(Qz.T, x)   # (10, N) x (N, 784) = (10, 784)

        self.pi = (weights_sum / (weights_sum.sum() + 10 * eps) + eps)
        self.mu = weighted_x_sum / (weights_sum[:, np.newaxis] + 10 * eps)

    def _log_likelihood(self, x):
        log_likelihood = np.zeros((x.shape[0], self.num_classes))   # (N, 10)

        for i in range(self.num_classes):
            log_likelihood[:, i] = np.sum(x * np.log(self.mu[i, :].clip(min=1e-50)), axis=1) + \
                                np.sum((1 - x) * np.log((1 - self.mu)[i, :].clip(min=1e-50)), axis=1)

        return log_likelihood

    def score_samples(self, x):
        lpr = self._log_likelihood(x) + np.log(self.pi)
        log_likelihoods = np.logaddexp.reduce(lpr, axis=1)              # (N, 10)
        Qz = np.exp(lpr - log_likelihoods[:, np.newaxis])               # P(Z|X ; mu, pi)

        return log_likelihoods, Qz

    def predict(self, x):
        print("\n[+] Predicting ...")
        predictions = np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            if i % 1000 == 0:
                print("[*] Predicting %dth image ..." % i)

            likelihood = np.zeros(10)

            for k in range(10):
                likelihood[k] = self.pi[k] * bernoulli_likelihood(self.mu[k, :], x[i])

            predictions[i] = np.argmax(likelihood)

        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='specify the data file', default='data')
    parser.add_argument('--num_classes', type=int, help='specify the number of classes', default=10)
    args = parser.parse_args()

    # Load dataset
    train_images, train_labels, test_images, test_labels = dataset.load_mnist_dataset(args.dir)

    train_images_binary = np.where(train_images > 0.5, 1, 0)

    # Learning the latent parameters
    model = Bernoulli_EM(args.num_classes, num_iteration=10, verbose=True)
    model.fit(train_images_binary)

    # plot_means(model.means)

    # Using learned latent to classify the test_images
    test_images_binary  = np.where(test_images > 0.5, 1, 0)
    predictions = model.predict(test_images_binary)

    # Clustering
    clusters = clustering(predictions, test_labels)

    clusters_label = np.zeros(10)
    for i in range(10):
        # print(i, ":", clusters[i])
        clusters_label[i] = np.argmax(clusters[i])
    # print(clusters_label)

    # print the confusion matrix
    TP = TN = FP = FN = 0

    for k in range(10):
        for index, item in enumerate(predictions):
            if clusters_label[int(item)] == k:
                if test_labels[index] == k:
                    TP += 1
                else:
                    FP += 1
            else:
                if test_labels[index] == k:
                    FN += 1
                else:
                    TN += 1

    print("\nconfusion matrix:")
    print("True Positive:", TP)
    print("True Negative:", TN)
    print("False Positive:", FP)
    print("False Negative:", FN)
    print("Sensitivity:", TP / (TP + FN))
    print("Specificity:", TN / (TN + FP))
