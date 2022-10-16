import os
import argparse
from emonet.run import model_classes, run

def entry():
    parser = argparse.ArgumentParser(
        usage='python -m emonet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', choices=model_classes.keys(), default='BaseGRU',
        help='The model to run.')
    parser.add_argument(
        '-s', '--setup', choices=["emotions"], type=str, default="emotions",
        help='Which setup to run.')
    flags = parser.parse_args()
    run({}, flags.model, flags.setup)


if __name__ == '__main__':
    print(os.getcwd())
    # os.chdir("C:/Users/Rylen/SFU/CMPT 713/emonet")
    entry()