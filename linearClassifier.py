# Created by Ethan Edwards on 11/21/2024
import os

import cv2
import math
import random
import numpy as np
import pandas as pd


def open_images(path):
    # Opens the csv files and extracts the images from them and returns them
    images = []
    data = pd.read_csv(path)
    headers = data.columns.values

    labels = data[headers[0]]
    labels = labels.values.tolist()

    pixels = data.drop(headers[0], axis=1)

    for i in range(0, data.shape[0]):
        row = pixels.iloc[i].to_numpy()
        grid = np.reshape(row, (28, 28))
        images.append(grid)
    return labels, images


def getBlWhImage(image):
    # Turns the image into black and white
    pixels = []
    for x in range(28):
        for y in range(28):
            if (int(image[x][y]) > 128):
                pixels.append(1)
            else:
                pixels.append(0)
    return np.reshape(pixels, (28, 28))


def verticalIntersections(image):
    # Gets the number of vertical intersections in black and white image
    counts = []
    prev = 0
    for y in range(28):
        count = 0
        for x in range(28):
            current = int(image[x][y])
            if (prev != current):
                count += 1
            prev = current
        counts.append(count)
    average = sum(counts)/28
    maximum = max(counts)
    return average, maximum


def horizontalIntersections(image):
    # Gets the number of horizontal intersections in black and white image
    counts = []
    for x in range(28):
        count = 0
        prev = 0
        for y in range(28):
            current = int(image[x][y])
            if (prev != current):
                count += 1
            prev = current
        counts.append(count)
    average = sum(counts)/28
    maximum = max(counts)
    return average, maximum


def calculateDensity(image):
    # calculates the density
    count = 0
    for x in range(28):
        for y in range(28):
            count = count + int(image[x][y])
    return count / (28 * 28)


def calc_symmetry(image):
    """
    Calculates the degree of symmetry
    """
    image = np.array(image)
    reflected_image = np.fliplr(image)
    xor_result = np.bitwise_xor(image, reflected_image)
    symmetry_measure = np.mean(xor_result)
    return symmetry_measure


def centroid_distance_variance(image):
    """
    Calculate the variance of distances from the centroid to the "on" pixels.
    """
    image = np.array(image)
    on_pixels = np.argwhere(image > 0)  # Get indices of all "on" pixels
    centroid = np.mean(on_pixels, axis=0)  # Compute centroid
    distances = np.linalg.norm(on_pixels - centroid, axis=1)  # Euclidean distances
    return np.var(distances)


def aspect_ratio(image):
    """
    Calculate the aspect ratio of the bounding box surrounding the digit.
    """
    binary_img = (np.array(image) > 0).astype(np.uint8)
    coords = np.column_stack(np.where(binary_img > 0))  # Get coordinates of non-zero pixels
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    return height / width if width > 0 else 0  # Avoid division by zero


def calc_euler(image):
    """
    Calculation of the Euler number for a binary image.
    """
    # Ensure the image is binary
    binary_image = (image > 0).astype(int)

    # Pad the image to handle edges
    padded_image = np.pad(binary_image, pad_width=1, mode='constant', constant_values=0)

    # Extract 2x2 blocks using slicing
    block_1 = padded_image[:-1, :-1]  # Top-left
    block_2 = padded_image[:-1, 1:]  # Top-right
    block_3 = padded_image[1:, :-1]  # Bottom-left
    block_4 = padded_image[1:, 1:]  # Bottom-right

    # Sum pixel values in each 2x2 block
    block_sums = block_1 + block_2 + block_3 + block_4

    # Count occurrences of 1s and 3s in the blocks
    n1 = np.sum(block_sums == 1)  # Single foreground pixel
    n3 = np.sum(block_sums == 3)  # Three foreground pixels

    # Compute Euler number
    euler_number = (n1 - n3) // 4
    return euler_number


def diag_symmetry(image):
    """
    Calculates the diagonal symmetry of a 2D grayscale image.
    """
    image = np.array(image)
    # Flip vertically and transpose
    flipped_image = np.flipud(image).T

    # Compute pixel-wise absolute difference between the image and its flipped version
    diff = np.abs(image - flipped_image)

    # Calculate the average symmetry score
    symmetry_score = 1 - (np.sum(diff) / (255 * image.size))  # Normalize to range [0, 1]

    return symmetry_score


def loop_count(image):
    """
    Calculate the number of loops in a binary image of a digit.
    """
    # Ensure the image is in the correct binary format
    binary_image = (image > 0).astype(np.uint8) * 255  # Convert to 0 and 255

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of closed contours (loops)
    count = 0
    for contour in contours:
        # Check if the contour is closed
        if cv2.isContourConvex(contour) is False:
            count += 1

    return count


def feature_extractor(images, weight, static_label, static=True, labels=None):
    """
    Extract features from the images.
    """
    features = []
    for image in images:
        # Standard features
        blackWhite = getBlWhImage(image)
        average_vert, max_vert = verticalIntersections(blackWhite)
        average_horiz, max_horiz = horizontalIntersections(blackWhite)
        density = calculateDensity(blackWhite)
        symmetry = calc_symmetry(blackWhite)

        # Custom features
        centroid_var = centroid_distance_variance(blackWhite)
        ar = aspect_ratio(blackWhite)
        euler = calc_euler(blackWhite)
        # diag = diag_symmetry(blackWhite)
        loop = loop_count(blackWhite)

        # Combine features
        feature_row = [
            density, symmetry, max_vert, average_vert, max_horiz, average_horiz,
            centroid_var, ar, euler, loop, weight
        ]
        # Static labeling for train/validation sets
        if static:
            feature_row.append(static_label)
        else:  # Dynamic labeling for test set
            feature_row.append(labels.pop(0))
        features.append(feature_row)
    return features


def dataloaderStatic(filename):
    """
    Handles loading data into a feature set from a file. Static version for loading train/validation sets
    """
    labels, images = open_images(filename)
    features = feature_extractor(images, -1, labels[0])
    return features


def dataloaderDynamic(filename):
    """
    Handles loading data into a feature set from a file
    """
    labels, images = open_images(filename)
    features = feature_extractor(images, -1, labels[0], static=False, labels=labels)
    return features


class LinearClassifier:
    def __init__(self):
        self.prior_probs = {}  # P(C = d)
        self.conditional_probs = {}  # P(F = x | C = d)

    def train(self, train_data, val_data):
        # Separate features and labels
        features = [row[:-2] for row in train_data]  # Exclude weight and label
        labels = [row[-1] for row in train_data]  # Label is the last element

        classes = set(labels)
        feature_count = len(features[0])

        # Compute prior probabilities
        self.prior_probs = {c: labels.count(c) / len(labels) for c in classes}

        # Initialize parameters for Gaussian modeling (mean, variance)
        self.gaussian_params = {c: [{} for _ in range(feature_count)] for c in classes}

        for c in classes:
            class_features = [features[i] for i in range(len(features)) if labels[i] == c]
            for j in range(feature_count):
                feature_values = [row[j] for row in class_features]
                # Store mean and variance for continuous features
                self.gaussian_params[c][j] = {
                    "mean": np.mean(feature_values),
                    "std": np.std(feature_values) + 1e-6  # Avoid division by zero
                }

    def test(self, test_data):
        features = [row[:-2] for row in test_data]  # Exclude weight and label
        labels = [row[-1] for row in test_data]

        correct = 0
        correct_digits = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0-9
        for i, instance in enumerate(features):
            # Compute posterior probabilities for each class
            class_probs = {}
            for c in self.prior_probs:
                log_prob = math.log(self.prior_probs[c])
                for j, feature_value in enumerate(instance):
                    # Gaussian model for continuous features
                    params = self.gaussian_params[c][j]
                    mean, std = params["mean"], params["std"]
                    likelihood = (1 / (std * np.sqrt(2 * np.pi))) * \
                                 np.exp(-((feature_value - mean) ** 2) / (2 * std ** 2))
                    log_prob += math.log(likelihood + 1e-6)  # Add epsilon to avoid log(0)
                class_probs[c] = log_prob

            # Assign the class with the maximum probability
            predicted_label = max(class_probs, key=class_probs.get)
            if predicted_label == labels[i]:
                correct += 1
                correct_digits[labels[i]] += 1

            # DEBUG
            # print(f"Instance {i + 1}: Predicted = {predicted_label}, Actual = {labels[i]}")

        # Report accuracy
        accuracy = correct / len(test_data)
        print(f"Total accuracy: {accuracy * 100:.2f}%")
        for i in range(10):
            acc_dig = correct_digits[i] / labels.count(i)
            print(f"Digit {i}: {acc_dig * 100:.2f}%")


def main():
    # Main function
    print("Starting...")
    # Load the training data
    train_path = "./dataset/train"
    test_path = "./dataset/test1.csv"
    val_path = "./dataset/val"

    # Confirm that the paths are valid
    if not os.path.exists(train_path):
        print("Invalid training path.")
        return

    if not os.path.exists(test_path):
        print("Invalid test path.")
        return

    if not os.path.exists(val_path):
        print("Invalid validation path.")
        return

    # Data
    TRAIN = []
    VAL = []
    # Weights
    WEIGHTS = []

    print("Loading training data...")
    for file in os.listdir(train_path):
        TRAIN += dataloaderStatic(train_path + "/" + file)

    print("Loading validation data...")
    for file in os.listdir(val_path):
        VAL += dataloaderStatic(val_path + "/" + file)

    print("Train size: " + str(len(TRAIN)))
    print("Val size: " + str(len(VAL)))

    # Train the model
    print("Training the model...")
    model = LinearClassifier()
    model.train(TRAIN, VAL)

    # Load the test data
    print("Loading test data...")
    TEST = dataloaderDynamic(test_path)

    # Test the model
    print("Testing the model...")
    model.test(TEST)

    # print(f"Total test errors: {errors}")
    print("Test complete.")

    # Custom test
    print("Would you like to test a custom file? (y/n)")
    response = input()
    if response.lower() == "y":
        print("Enter the path to the custom file:")
        custom_path = input()
        if os.path.exists(custom_path):
            print("Loading custom data...")
            CUSTOM = dataloaderDynamic(custom_path)
            print("Testing the custom data...")
            model.test(CUSTOM)
        else:
            print("Invalid path.")


if __name__ == "__main__":
    main()
