"""
Netflix benchmark.
"""


config_ = {
    "train_files": [0],
    "train_test_split": 0.8,
    "evaluation": "mse",
    "models": [
        {
            "method": "TrainVanillaDVAE",
            "method_config": {
                "encoding_dims": [256, 128],
                "latent_dim": 32,
                "activation": "relu",
                "optimizer": {"Adam": {"learning_rate": 0.001}},
                "loss": {"MeanSquaredError": {}},
            },
            "train_config": {
                "batch_size": 32,
                "n_batches": 1000,
                "mask_rate": 0.8
            }
        }
    ]
}


def main():

    import argparse as ap
    import json

    from recommenders import train
    from recommenders.netflix.process import Process

    parser = ap.ArgumentParser(description='Benchmarking recommender systems '
                                           'with the Netflix Prize data.')
    parser.add_argument('--data-dir', dest='data_dir', action='store',
                        default=None, help='Data directory.')
    parser.add_argument('--config', dest='config', action='store',
                        default=None, help='Configuration file location.')

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        config_.update(config)

    netflix_batcher = Process(args.data_dir, config_["train_test_split"])
    netflix_batcher.load_training(config_["train_files"])

    for c in config_["models"]:
        c["method_config"]["input_dim"] = netflix_batcher.input_dim
        model = getattr(train, c["method"])(c["method_config"])
        model.train(netflix_batcher, c["train_config"])


if __name__ == "__main__":

    main()

