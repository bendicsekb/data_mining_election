# 2AMM20 Research Topic in Data Mining Group 3
import heapq
import sys
import pandas as pd

import data_refinement
import beam_search

# Dataset parameters

PATH = "datasets/student-mat.csv"  # file path to the dataset
targets = ["G1", "G2", "G3"]  # column names of the target attributes
descriptors = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "freetime", "goout", "Dalc", "Walc", "health", "absences"]  # column names of the descriptor attributes

# Beam search parameters (all integers)
w = 10  # None  # beam width
d = 4  # None  # search depth
b = 4  # None  # static binning bin size
q = 10  # None  # top q subgroups to return


def print_setup():
    print("Running top-q beam search on dataset", PATH)
    print("With target attributes: ", targets)
    print("and descriptor attributes: ", descriptors)
    print("Beam search parameters:\n\tBeam width ", w, "\n\tSearch depth ", d, "\n\tNumber of bins ", b, "\n\tNumber of subgroups returned ", q, "\n")


def print_result(result: list[(float, int, data_refinement.Description)]):
    print("\nFound subgroups")
    for i in reversed(range(len(result))):
        sys.stdout.write(str(i+1))
        sys.stdout.write(": ")
        heapq.heappop(result)[2].print_description()


if __name__ == '__main__':
    if PATH == "":
        PATH = str(input("Enter the file path to the dataset: "))
    data = pd.read_csv(PATH, delimiter=";")

    print("Please check if the data got read in correctly, if not change the read_csv parameters")
    print(data.head())

    if len(targets) == 0:
        temp = str(input("Please supply the target attributes in the format: attribute1, attribute2, .., attributex\n"))
        targets = [s.strip() for s in temp.split(",")]
    if len(descriptors) == 0:
        columns = data.columns
        descriptors = [attr for attr in columns if attr not in targets]  # descriptors != targets

    if w is None:
        w = int(input("Enter the beam width size (integer): "))
    if d is None:
        d = int(input("Enter the beam search depth (integer): "))
    if b is None:
        b = int(input("Enter the number of bins (integer): "))
    if q is None:
        q = int(input("Enter the number of results to be returned (integer): "))

    print_setup()
    dataset = data_refinement.DataSet(data, targets, descriptors)
    result = beam_search.beam_search(w, d, b, q, dataset)
    print_result(result)



