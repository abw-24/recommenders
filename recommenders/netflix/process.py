"""
Process Netflix data for benchmarking.
"""


import os
import numpy as np
import tensorflow as tf


class Process(object):

    TRAIN_FILENAMES = ["combined_data_{}".format(i) for i in range(1, 5)]
    TITLE_FILENAME = "movie_titles.csv"
    VALIDATION_FILENAME = "probe.txt"

    MOVIE_DIM = 17769

    def __init__(self, data_dir, train_test_split=0.8):
        """
        Process Netflix movie data into various forms for recommender problems.
        Utilizes a dictionary-style sparse storage format upon read in to conserve memory
        usage both for the raw data and to avoid any explicit groupBy operations.
        :param data_dir: Data directory
        :param train_test_split: Train/test split rate (value specifies train)
        """

        self._dir = data_dir
        self._split_rate = train_test_split
        self._training_paths = [os.path.join(self._dir, t) for t in self.TRAIN_FILENAMES]
        self._validation_path = os.path.join(self._dir, self.VALIDATION_FILENAME)

        self._train = {}
        self._current_movie = None
        self._n_users = 0
        self._train_map = {}
        self._test_map = {}

    def _parse_training_line(self, line):
        """
        Parse a single line of training data into sparse format.
        :param line: Single line (string) of raw training data
        :return: Array of [date, sparse vector tuple]
        """

        line_elements = line.split(",")

        if len(line_elements) == 1:
            # make sure to offset the movie ID, as it's indexed from 1
            self._current_movie = int(line_elements[0][:-1]) - 1
            return None

        date = line_elements[2]
        sparse_vec = (
            line_elements[0],
            {
                "indices": [self._current_movie],
                "values": [int(line_elements[1])]
            }
        )

        return date, sparse_vec

    def load_training(self):
        """
        Iterate over training data files and load line-by-line.
        """

        for file in self._training_paths:

            with open(file, "rb") as f:

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
                        self._train[user_id]["indices"].append(sparse_vec["indices"])
                        self._train[user_id]["values"].extend(sparse_vec["values"])

                    except KeyError:
                        self._train[user_id] = {
                            "indices": [sparse_vec["indices"]],
                            "values": sparse_vec["values"]
                        }

                        # flip a coin to decide if the new id goes in the
                        # train or test id lookups
                        if np.random.binomial(1, self._split_rate):
                            self._train_map[self._n_users] = user_id
                        else:
                            self._test_map[self._n_users] = user_id
                        self._n_users += 1


    def masked_batch(self, batch_size=32, mask_rate=0.5):
        """
        Use the user map and sparse formatted training data to grab
        a random batch, mask, and return lists of both the original
        and masked batches for training.
        :param batch_size: Batch size
        :param mask_rate: Mask Rate
        :return: Array of [masked_batch, target_batch]
        """

        masked = []
        target = []
        batch_len = 0
        batch_keys = []

        # take a random set of batch keys
        while batch_len < batch_size:
            batch_index = np.random.randint(0, self._n_users)
            try:
                batch_key = self._train_map[batch_index]
            except KeyError:
                continue
            else:
                batch_keys.append(batch_key)
                batch_len += 1

        for k in batch_keys:

            indices = self._train[k]["indices"]
            values = self._train[k]["values"]

            # for the target, just append a sparse tensor of the true data
            masked.append(tf.SparseTensor(indices, values, [self.MOVIE_DIM]))

            # for the masked input, flip a coin based on the `noise` level to determine
            # if each value is masked, and create a sparse tensor
            masked_values = [-1 if np.random.binomial(1, mask_rate) else v for v in values]
            masked.append(tf.SparseTensor(indices, masked_values, [self.MOVIE_DIM]))

        return masked, target
