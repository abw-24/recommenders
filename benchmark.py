"""
End-to-end benchmarking (Netflix data processing, training, evaluation).
"""


config_ = {
    "train_files": [0, 1],
    "train_test_split": 0.8,
    "models": [
        {
            "method": "TrainVanillaDAE",
            "method_config": {
                "encoding_dims": [512, 128],
                "latent_dim": 32,
                "activation": "sigmoid",
                "optimizer": {"Adam": {"learning_rate": 0.001}},
                "loss": {"MeanAbsoluteError": {}},
                "activity_regularizer": {"L1": {"l1": 0.005}},
                "sparse_flag": False,
                "batch_size": 64,
                "n_batches": 500,
                "mask_rate": [0.1, 0.5],
                "mask_schedule": "increasing",
                "n_evaluations": 100
            }
        }
    ]
}


def main():

    import argparse as ap
    import json

    from recommenders import train
    from recommenders.netflix.process import Process

    parser = ap.ArgumentParser(description='Benchmarking recommender systems'
                                           ' with the Netflix Prize data.')
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
        model = getattr(train, c["method"])(netflix_batcher, c["method_config"])
        model.train()


if __name__ == "__main__":

    main()

