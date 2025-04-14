import sys
import os


def parse_args(argv):
    args = {
        "input": None,
        "bits": None,
        "buffer_size": None
    }
    
    if (len(argv) == 0):
        print("Please provide input file type -h for help")

    i = 0
    while i < len(argv):
        if (argv[i] == '-h'):
            print("usage: python3 branchPrediction.py <input file> <bits to use (defaults to 0)> <size of the branch-prediction buffer>")

    if not os.path.isfile(argv[1]):
        print("File not found")


    return args



def branch_prediction():
    parse_args(sys.argv)


if __name__ == '__branch_prediction__':
    branch_prediction()


