import sys
import argparse
from dvae.learning_algo import LearningAlgorithm

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument('--input', type=str, default=None, help='input data')
        self.parser.add_argument('--output_dir', type=str, default=None, help='output file path')
        self.parser.add_argument('-l', type=float, default=None, help='learning rate')
        self.parser.add_argument('-b', type=int, default=None, help='batch size')
        self.parser.add_argument('-n', type=int, default=None, help='number of epochs')
        self.parser.add_argument('--xdim', type=int, default=None, help='dimension of input data')
        self.parser.add_argument('--zdim', type=int, default=None, help='dimension of latent variable data')

    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params

if __name__ == '__main__':
    params = Options().get_params()
    learning_algo = LearningAlgorithm(params=params)
    learning_algo.train()

