import numpy as np
from datasets.cifar import cifar10
from tensorflow.examples.tutorials.mnist import input_data
from datasets.stl10 import stl10
import helpers

class Dataset(object):

    def __init__(self, params):
        """
        Load dataset and reshape according to image dimensions
        """
        self.x_train = None
        self.x_validation = None
        self.x_test = None

        self.params = params

    def mnist(self):
        # os.path.expanduser("~")
        mnist = input_data.read_data_sets(self.params["data_dir"], reshape=False)

        X_train, Y_train           = mnist.train.images, mnist.train.labels
        X_validation, Y_validation = mnist.validation.images, mnist.validation.labels
        X_test, Y_test             = mnist.test.images, mnist.test.labels

        # Get image shape
        print("Image Shape: {}".format(X_train[0].shape))
        print()
        print("Training Set:   {} samples".format(len(X_train)))
        print("Validation Set: {} samples".format(len(X_validation)))
        print("Test Set:       {} samples".format(len(X_test)))

        # Zero Padding
        X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        print("Updated Image Shape: {}".format(X_train[0].shape))

        prod = np.prod(self.params["image_dims"])

        self.x_train = np.reshape(X_train,(-1, prod))
        self.x_validation = np.reshape(X_validation,(-1, prod))
        self.x_test = np.reshape(X_test,(-1, prod))

        return self.x_train, self.x_test

    def cifar(self):
        cifar10.data_path = self.params["data_dir"]

        X_train, Y_train, _           = cifar10.load_training_data()
        X_test, Y_test, _             = cifar10.load_test_data()

        print("Image Shape: {}".format(X_train[0].shape))
        print()
        print("Training Set:   {} samples".format(len(X_train)))
        print("Test Set:       {} samples".format(len(X_test)))

        prod = np.prod(self.params["image_dims"])

        self.x_train = np.reshape(X_train,(-1, prod))
        self.x_test = np.reshape(X_test,(-1, prod))

        return self.x_train, self.x_test

    def stl10(self, colors="rgb"):
        if colors == "rgb":
            return stl10.get_images()/255.
        elif colors == "ycbr":
            return helpers.rgb2ycbcr(stl10.get_images())/255.

    def set14(self, scale):
        """
        Returns an array of grayscale images
        """
        images = []

        for i in range(1,14):
            images.append(
                helpers.load_input_image("{}/{}/img_{:03d}.png".format(self.params["data_dir"], self.params["dataset"], i),
                scale=scale)/255.
            )

        return np.array(images)
