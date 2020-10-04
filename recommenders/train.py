"""
Top level training classes
"""

from nets import dense, train


class TrainVanillaDVAE(object):
    """
    Train a vanilla VAE from the `nets` repo.
    """

    def __init__(self, method_config):

        self._network_config = method_config
        self._model = dense.DenseVAE(self._network_config)

        self._compiled_model = train.model_init(
            self._model,
            self._network_config["loss"],
            self._network_config["optimizer"],
            (None, self._network_config["input_dim"])
        )

    def train(self, data_obj, train_config):
        """
        Takes the Netflix data object and a training config and trains
        to de-noise the masked ratings.
        and trains the VAE.
        :return:
        """

        batch_size = train_config["batch_size"]
        mask_rate = train_config["mask_rate"]

        for i in range(train_config["n_batches"]):

            masked, target = data_obj.masked_batch("train", batch_size, mask_rate)
            loss_, grads = train.grad(self._compiled_model, masked, target)
            updates = zip(grads, self._compiled_model.trainable_variables)
            self._compiled_model.optimizer.apply_gradients(updates)

            if i % 10 == 0:
                print("Batch {b} loss: {l}".format(b=i, l=loss_))
