## TODO: Pull the random data generation out from the Environment.py file and have it run here instead.

class DataGen:
    def __init__(self, seed=None):
        self.rng=np.random.default_rng(seed)

    def generate_games(self, num_games):
        """
        Batch-generation of poker hands under one seed
        """