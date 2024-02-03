import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def load_data():
    """
    load the raw data, convert each line of digits into an array of 'pixels'
    also normalize to [0, 1] by dividing by 6
    """
    raw_data = []
    with open('./ProjectDigits_materials/mfeat-pix.txt', 'r') as f:
        for line in f:
            pixels = np.fromstring(line, dtype=int, sep=' ')
            raw_data.append(pixels)
    return np.array(raw_data) / 6


def n_image_in_category(data) -> int:
    """
    count the number of images in each category, assuming there's an even number of images in each class.
    """
    return int(data.shape[0] / 10)


def train_test_split(raw_data: np.array):
    """
    split data into training and testing data.
    use an even split by using the first 50% of the images for each class,
    assuming there's an even number of images in each class, there are 10 classes and the images are sorted by class
    """
    n_images = n_image_in_category(raw_data)
    split = int(n_images / 2)

    split_indices = [np.arange(label * n_images, label * n_images + split) for label in range(10)]

    split_indices = np.array(split_indices).flatten()
    train_data = raw_data[split_indices]
    test_data = np.delete(raw_data, split_indices, axis=0)

    # sanity checks, disable if necessary
    assert (n_images == 200 and split == 100)
    assert (train_data.shape == test_data.shape == (1000, 240))

    return train_data, test_data


def one_hot_y(X: np.array):
    """
    return one-hot encoded class labels, under the default assumptions
    """
    n_images = n_image_in_category(X)
    y = np.zeros((X.shape[0], 10), dtype=np.int64)
    for i in range(10):
        for j in range(n_images):
            y[i * n_images + j, i] = 1
    return y


def sample_images_for_each_digit(data, n_samples=5) -> list:
    """
    return n_samples of each class
    """
    n_images = n_image_in_category(data)
    sample_images = []
    for digit in range(10):
        sample_indices = np.random.choice(n_images, n_samples, replace=False)
        sample = data[digit * n_images + sample_indices]
        for img in sample:
            sample_images.append(img)
    return sample_images


def plot_sample_for_each_digit(data, n_samples=5, figsize=(10, 20), horizontal=False):
    fig = plt.figure(figsize=figsize)
    sample_images = sample_images_for_each_digit(data, n_samples)

    for i in range(len(sample_images)):
        row, col = (10, n_samples) if not horizontal else (n_samples, 10)
        fig.add_subplot(row, col, i + 1)

        img = sample_images[i].reshape(16, 15)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')


def perform_linear_ridge_regression(X, y, tiny=0.00001):
    """
    Simple implementation of Ridge regression
    """
    w_prime = np.linalg.inv((X.T @ X) + tiny * np.identity(X.shape[1])) @ X.T @ y
    w_opt = w_prime.T
    return w_opt


def evaluate_w_opt(w_opt: np.array,
                   train_data: np.array, test_data: np.array,
                   y_train: np.array, y_test: np.array,
                   verbose=True):
    """
    Evaluate by calculating the misclassification rate on training and test set
    """
    misclassified_train = 0
    misclassified_test = 0

    for i in range(test_data.shape[0]):
        prediction = w_opt @ train_data[i]

        if np.argmax(prediction) != np.argmax(y_train[i]):
            misclassified_train += 1

        prediction = w_opt @ test_data[i]
        if np.argmax(prediction) != np.argmax(y_test[i]):
            misclassified_test += 1

    if verbose:
        print(
            f"train: misclassified: {misclassified_train}, misclassification rate: {misclassified_train / y_train.shape[0] * 100:.1f}%")
        print(
            f"test: misclassified: {misclassified_test}, misclassification rate: {misclassified_test / y_test.shape[0] * 100:.1f}%")
    return misclassified_train, misclassified_test


def train_val_split(training_data: np.array, y: np.array, n_folds: int, fold: int):
    """
    split training data into training and validation set.
    Used for k-fold cross validation. Set n_folds for K and fold for current fold.
    """
    n_images_per_class = n_image_in_category(training_data)
    n_validation_images = int(n_images_per_class / n_folds)

    # images per class / number of folds should result in a whole integer
    # leave-one-out means number of folds == images per class
    assert n_validation_images == n_images_per_class / n_folds

    indices_offset = n_validation_images * fold
    val_indices = [list(
        range(i * n_images_per_class + indices_offset, i * n_images_per_class + indices_offset + n_validation_images)
    ) for i in range(10)]
    val_indices = np.array(val_indices).flatten()
    return (np.delete(training_data, val_indices, axis=0), training_data[val_indices],
            np.delete(y, val_indices, axis=0), y[val_indices])


def pca_ridge_regression(train_set: np.array, val_set: np.array, train_y: np.array, val_y: np.array, n_components: int):
    pca: PCA = PCA(n_components=n_components, svd_solver='full')
    fitted = pca.fit(train_set)
    train_pca = fitted.transform(train_set)
    validation_pca = fitted.transform(val_set)

    w_opt = perform_linear_ridge_regression(train_pca, train_y)
    result = evaluate_w_opt(w_opt, train_pca, validation_pca, train_y, val_y, verbose=False)

    # calculate misclassification rates for train and val splits
    misclassification_rates = np.array(result) / val_y.shape[0]
    return misclassification_rates


def k_fold_analysis(train_data: np.array, y: np.array, n_folds: int, func):
    """
    perform K fold analysis on a given func.
    func should take in train_set, val_set, train_y, val_y and return misclassification rates on train and val sets
    returns misclassification rates on train and validation sets for every fold
    """
    k_fold_results = np.zeros((n_folds, 2))
    for k in range(n_folds):
        train_set, val_set, train_y, val_y = train_val_split(train_data, y, n_folds, k)

        misclassification_rates = func(train_set, val_set, train_y, val_y)

        k_fold_results[k] = misclassification_rates
    return k_fold_results
