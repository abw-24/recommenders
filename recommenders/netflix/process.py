"""
Process Netflix data for training/evaluation.
"""


import os
import numpy as np
import tensorflow as tf


class Process(object):

    TRAIN_FILENAMES = ["combined_data_{}.txt".format(i) for i in range(1, 5)]
    TITLE_FILENAME = "movie_titles.csv"
    VALIDATION_FILENAME = "probe.txt"

    MOVIE_DIM = 17769

    def __init__(self, data_dir, train_test_split=0.8):
        """
        Process Netflix movie data (as available via kaggle)for recommender problems.
        Utilizes a dictionary sparse storage format upon read-in to conserve memory.
        :param data_dir: Data directory
        :param train_test_split: Train/test split rate (value specifies train)
        """

        self._dir = data_dir
        self._split_rate = train_test_split
        self._training_paths = [os.path.join(self._dir, t) for t in self.TRAIN_FILENAMES]
        self._validation_path = os.path.join(self._dir, self.VALIDATION_FILENAME)

        self._labeled_data = {}
        self._current_movie = None
        self._n_movies = 0
        self._n_users = 0
        self._train_map = {}
        self._test_map = {}

    def _parse_training_line(self, line):
        """
        Parse a single line of training data into sparse format.
        Normalizes ratings by dividing everything by 5.
        :param line: Single line (string) of raw training data
        :return: Array of [date, sparse vector tuple]
        """

        line_elements = line.strip("\n").split(",")

        if len(line_elements) == 1:
            # make sure to offset the movie ID by 1, as it's indexed from 1
            self._current_movie = int(line_elements[0].split(":")[0]) - 1
            self._n_movies += 1
            return None

        date = line_elements[2]
        sparse_vec = (
            line_elements[0],
            {
                "indices": [self._current_movie],
                "values": [int(line_elements[1])/5.0]
            }
        )

        return date, sparse_vec

    def load_training(self, files=(0,1,2,3)):
        """
        Iterate over training data files and load line-by-line.
        """

        for i, file in enumerate([self._training_paths[f] for f in files]):

            with open(file, "r") as f:

                for line in f:

                    res = self._parse_training_line(line)
                    # if the result is None, we just hit a new movie
                    if res is None:
                        continue
                    else:
                        # otherwise we have parsed data
                        date, parsed_data = res
                        user_id, sparse_vec = parsed_data

                    try:
                        self._labeled_data[user_id]["indices"].extend(sparse_vec["indices"])
                        self._labeled_data[user_id]["values"].extend(sparse_vec["values"])

                    except KeyError:
                        # if we can't find the id, it's a new user.
                        self._labeled_data[user_id] = {
                            "indices": sparse_vec["indices"],
                            "values": sparse_vec["values"]
                        }

                        # flip a coin to decide if the new id goes in the
                        # train or test id lookups
                        if np.random.binomial(1, self._split_rate):
                            self._train_map[self._n_users] = user_id
                        else:
                            self._test_map[self._n_users] = user_id
                        self._n_users += 1

    def masked_batch(self, data="train", batch_size=32, mask_rate=0.2):
        """
        Use the user map and sparse formatted training data to grab
        a random batch, mask, and return lists of both the original
        and masked batches for training or evaluation.
        :param data: Specification of "train" or "test" batches
        :param batch_size: Batch size
        :param mask_rate: Mask Rate
        :return: Array of [masked_batch, target_batch]
        """

        batch_len = 0
        index_pairs = []
        values = []

        # construct batch
        while batch_len < batch_size:

            batch_index = np.random.randint(0, self._n_users)
            try:
                if data == "train":
                    k = self._train_map[batch_index]
                elif data == "test":
                    k = self._test_map[batch_index]
                else:
                    raise ValueError("Argument `data` can only take values 'train' or 'test'")
            except KeyError:
                continue
            else:
                index_pairs.extend([[batch_len, j] for j in self._labeled_data[k]["indices"]])
                values.extend(self._labeled_data[k]["values"])
                batch_len += 1

        # for the target, create a sparse tensor of the true data
        target = tf.SparseTensor(index_pairs, values, [batch_size, self._n_movies])

        # for the masked input, flip a coin based on the `noise` level to determine
        # if each value is masked, and then create the sparse tensor
        masked_values = [-0.5 if np.random.binomial(1, mask_rate) else v for v in values]
        masked = tf.SparseTensor(index_pairs, masked_values, [batch_size, self._n_movies])

        return masked, target

    @property
    def input_dim(self):
        return self._n_movies