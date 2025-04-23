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
        sys.exit(1)


    if (argv[1] == '-h'):
        print("usage: python3 branchPrediction.py <input file> <bits to use> <size of the branch-prediction buffer>")
        sys.exit(1)

    if not os.path.isfile(argv[1]):
        print("File not found")
    else:
        #print(f"Input file: {argv[1]}")
        args["input"] = argv[1]

    if not argv[2].isdigit():
        print("Bits must be an integer")
        sys.exit(1)
    else:
        bits = int(argv[2])
    if bits > 3:
        print("Please provide bits in range 0-3")
        sys.exit(1)
    else:
        args["bits"] = bits

    if not argv[3].isdigit():
        print("buffer size must be an integer")
        sys.exit(1)
    else:
        args["buffer_size"] = argv[3]


    return args



def branch_prediction():
    args = parse_args(sys.argv)

    # should probably be (2 * counter_size) - 1
    counter_size = int(args["bits"])
    buffer_size = int(args["buffer_size"])
    filename = args["input"]
    file = open(filename, 'r')


    if counter_size > 0:
        N = buffer_size // counter_size
        counters = [0] * N

    correct_bp = 0
    total_bp = 0

    match counter_size:
        case 0:
            for line in file:
                total_bp += 1
                prediction = 0
                parts = line.split()
                if int(parts[1]) == prediction:
                    correct_bp += 1
        case 1:
            for line in file:
                total_bp += 1
                parts = line.split()
                A = int(parts[0],16)
                branch = int(parts[1])
                index = A % N
                if counters[index] == branch:
                    correct_bp += 1
                else:
                    counters[index] = branch
        case 2:
            for line in file:
                total_bp += 1
                parts = line.split()
                A = int(parts[0],16)
                branch = int(parts[1])
                index = A % N
                if counters[index] in [2, 3] and branch == 1:
                    counters[index] = min(counters[index] + 1, 3)
                    correct_bp += 1
                elif counters[index] in [0, 1] and branch == 0:
                    counters[index] = max(counters[index] - 1, 0)
                    correct_bp += 1
                elif counters[index] in [0, 1] and branch == 1:
                    counters[index] = min(counters[index] + 1, 3)
                elif counters[index] in [2, 3] and branch == 0:
                    counters[index] = max(counters[index] - 1, 0)
        case 3:
            for line in file:
                total_bp += 1
                parts = line.split()
                A = int(parts[0],16)
                branch = int(parts[1])
                index = A % N

                #print(f"counter {counters[index]} compared to branch {branch}")

                if counters[index] in [3, 4, 5] and branch == 1:
                    #print("branch")
                    counters[index] = min(counters[index] + 1, 5)
                    correct_bp += 1
                elif counters[index] in [0, 1, 2] and branch == 0:
                    #print("no branch")
                    counters[index] = max(counters[index] - 1, 0)
                    correct_bp += 1
                elif counters[index] in [0, 1, 2] and branch == 1:
                    counters[index] = min(counters[index] + 1, 5)
                elif counters[index] in [3, 4, 5] and branch == 0:
                    counters[index] = max(counters[index] - 1, 0)

    percent_correct = round (100*(correct_bp/total_bp),2)
    print(percent_correct)
    #print(f"File              || #branches | {counter_size}-bit")
    #print(f"{filename} || {total_bp}    | {percent_correct} ")




if __name__ == '__main__':
    branch_prediction()


