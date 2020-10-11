"""
Top level training classes.
"""

from nets import dense, train
import tensorflow as tf
import numpy as np


class TrainVanillaDVAE(object):
    """
    Train a vanilla VAE from the `nets` repo.
    """

    def __init__(self, data_obj, method_config):

        self._data_obj = data_obj
        self._method_config = method_config

        self._batch_size = self._method_config["batch_size"]
        self._mask_range = self._method_config["mask_rate"]
        self._n_batches = self._method_config["n_batches"]
        self._mask_schedule = self._method_config["mask_schedule"]
        self._n_evals = self._method_config["n_evaluations"]

        self._model = dense.DenseVAE(self._method_config)

        self._compiled_model = train.model_init(
            self._model,
            self._method_config["loss"],
            self._method_config["optimizer"],
            (None, self._method_config["input_dim"])
        )

    def train(self):
        """
        Train for the configured number of batches and periodically evaluate on test.
        and trains the VAE.
        :return:
        """
        mask_delta = self._mask_range[1]-self._mask_range[0]

        for i in range(self._n_batches):

            if self._mask_schedule == "random":
                # random over range
                mask_rate = np.random.uniform(self._mask_range[0], self._mask_range[1])
            elif self._mask_schedule == "increasing":
                # linear interpolation between min and max of range
                mask_rate = self._mask_range[0] + (i/float(self._n_batches))*(mask_delta)
            else:
                raise ValueError("Currently only supporting 'random' and 'increasing'.")

            masked, target = self._data_obj.masked_batch("train", self._batch_size, mask_rate)
            # for now, convert sparse tensor to dense to get it working...
            #TODO: explore solutions for using SparseTensor directly
            loss_, grads = train.grad(
                    self._compiled_model,
                    tf.sparse.to_dense(masked),
                    tf.sparse.to_dense(target)
            )
            updates = zip(grads, self._compiled_model.trainable_variables)
            self._compiled_model.optimizer.apply_gradients(updates)

            if i % 10 == 0:
                self._evaluate()

    def _evaluate(self):
        """
        Evaluate on a set of test batches.
        :return:
        """

        test_loss = 0.0
        for i in range(self._n_evals):
            masked, target = self._data_obj.masked_batch("test", self._batch_size, 0.2)
            prediction = self._compiled_model(tf.sparse.to_dense(masked))
            test_loss += self._compiled_model.loss(tf.sparse.to_dense(target), prediction)

        print("Average Test Loss: {}".format(test_loss/self._n_evals))
