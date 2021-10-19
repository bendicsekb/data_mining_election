# 2AMM20 Research Topic in Data Mining Group 3
import heapq
import sys
import pandas as pd
import time
from tqdm import tqdm

import data_refinement
import beam_search

# party votes columns
parties_abs = ['VVD', 'D66', 'PVV (Partij voor de Vrijheid)', 'CDA', 'SP (Socialistische Partij)', 'Partij van de Arbeid (P.v.d.A.)', 'GROENLINKS', 'Forum voor Democratie', 'Partij voor de Dieren', 'ChristenUnie', 'Volt', 'JA21', 'Staatkundig Gereformeerde Partij (SGP)', 'DENK', '50PLUS', 'BBB', 'BIJ1', 'CODE ORANJE', 'NIDA', 'Splinter', 'Piratenpartij', 'JONG', 'Trots op Nederland (TROTS)', 'Lijst Henk Krol', 'NLBeter', 'Blanco (Zeven, A.J.L.B.)', 'LP (Libertaire Partij)', 'OPRECHT', 'JEZUS LEEFT', 'DE FEESTPARTIJ (DFP)', 'U-Buntu Connected Front', 'Vrij en Sociaal Nederland', 'Partij van de Eenheid', 'Wij zijn Nederland', 'Partij voor de Republiek', 'Modern Nederland', 'De Groenen']  # column names of the target attributes
parties_rel = ['VVD (%)', 'D66 (%)', 'PVV (Partij voor de Vrijheid) (%)', 'CDA (%)', 'SP (Socialistische Partij) (%)', 'Partij van de Arbeid (P.v.d.A.) (%)', 'GROENLINKS (%)', 'Forum voor Democratie (%)', 'Partij voor de Dieren (%)', 'ChristenUnie (%)', 'Volt (%)', 'JA21 (%)', 'Staatkundig Gereformeerde Partij (SGP) (%)', 'DENK (%)', '50PLUS (%)', 'BBB (%)', 'BIJ1 (%)', 'CODE ORANJE (%)', 'NIDA (%)', 'Splinter (%)', 'Piratenpartij (%)', 'JONG (%)', 'Trots op Nederland (TROTS) (%)', 'Lijst Henk Krol (%)', 'NLBeter (%)', 'Blanco (Zeven, A.J.L.B.) (%)', 'LP (Libertaire Partij) (%)', 'OPRECHT (%)', 'JEZUS LEEFT (%)', 'DE FEESTPARTIJ (DFP) (%)', 'U-Buntu Connected Front (%)', 'Vrij en Sociaal Nederland (%)', 'Partij van de Eenheid (%)', 'Wij zijn Nederland (%)', 'Partij voor de Republiek (%)', 'Modern Nederland (%)', 'De Groenen (%)']  # column names of the target attributes

# Synthetic ataset parameters
PATH = "datasets/"  # file path to the datasets folder
DEFAULT_FILE_NAME = "Demographic_and_election_dataset"  # file name without its file extension
STARTING_INDEX = 1  # starting suffix of the set of datasets
ENDING_INDEX = 2  # last suffix present in the set of datasets
TARGET_DESCRIPTION = None  # description of the generated interesting subgroup

# Target, descriptors, or unwanted descriptor definition
# If descriptors are empty, all attributes in the dataset, not in targets and not in unwanted_descriptors will be used
targets = []
descriptors = []  # column names of the descriptor attributes
unwanted_descriptors = []

# list(set().union(['RegioNaam', 'Region code', 'Kiesgerechtigden', 'Opkomst', 'OngeldigeStemmen', 'BlancoStemmen', 'GeldigeStemmen'], parties_abs))

# Beam search parameters (all integers)
w = 10  # None  # beam width
d = 1  # None  # search depth
b = 1  # None  # static binning bin size
q = 10  # None  # top q subgroups to return


def print_setup():
    print("\nRunning top-q beam search on datasets", DEFAULT_FILE_NAME, " at", PATH)
    print("From starting index", STARTING_INDEX, " to ending index", ENDING_INDEX)
    print("With target attributes: ", targets)
    print("and descriptor attributes: ", descriptors)
    print("\nTarget description is:")
    TARGET_DESCRIPTION.print_description()
    print("\nBeam search parameters:\n\tBeam width ", w, "\n\tSearch depth ", d, "\n\tNumber of bins ", b, "\n\tNumber of subgroups returned ", q, "\n")


def print_result(result: list[(float, int, data_refinement.Description)]):
    print("\nFound subgroups")
    for i in reversed(range(len(result))):
        sys.stdout.write(str(i+1))
        sys.stdout.write(": ")
        heapq.heappop(result)[2].print_description()


# Set target description
def set_target_description():
    target_description = data_refinement.Description()
    rule = data_refinement.Rule(data_refinement.RuleType.BINARY, "a1", "==", 1.0)
    target_description.add_rule(rule)
    return target_description


def process_result(top_q: list[(float, int, data_refinement.Description)]):
    if len(top_q) == 0:  # although it should not happen
        return -1

    top_q_sorted_list = sorted(top_q.copy(), key=lambda tup: tup[0], reverse=True)  # get top_q, sort descendlingly on quality
    description_list = [tup[2] for tup in top_q_sorted_list]  # Keep only the descriptions

    for desc in description_list:
        if desc.to_string() == TARGET_DESCRIPTION.to_string():
            return description_list.index(desc) + 1  # Return index + 1, as indices start at 0

    return -1


if __name__ == '__main__':
    if PATH == "" or PATH is None:
        PATH = str(input("Enter the file path to the dataset: "))
    if DEFAULT_FILE_NAME == "" or DEFAULT_FILE_NAME is None:
        DEFAULT_FILE_NAME = str(input("Enter the default data file without index: "))
    if STARTING_INDEX < 0 or STARTING_INDEX is None:
        STARTING_INDEX = int(input("Enter the starting index: "))
    if ENDING_INDEX is None:
        ENDING_INDEX = int(input("Enter the ending index: "))
    if ENDING_INDEX < STARTING_INDEX:
        print("Ending index", ENDING_INDEX, " is less than starting index", STARTING_INDEX, " stopping")
        raise Exception
    if TARGET_DESCRIPTION is None:
        TARGET_DESCRIPTION = set_target_description()

    data = pd.read_csv(PATH + DEFAULT_FILE_NAME + str(STARTING_INDEX) + ".csv", delimiter=",")
    print(data.head())
    input("Please check if the data got read in correctly\n"
          "if not then interrupt and change the parameters, or type any key to continue")

    if len(targets) == 0:
        temp = str(input("Please supply the target attributes in the format: attribute1, attribute2, .., attributex\n"))
        targets = [s.strip() for s in temp.split(",")]
    if len(descriptors) == 0:
        columns = data.columns
        descriptors = [attr for attr in columns if (attr not in targets and attr not in unwanted_descriptors)]  # descriptors != targets

    if w is None:
        w = int(input("Enter the beam width size (integer): "))
    if d is None:
        d = int(input("Enter the beam search depth (integer): "))
    if b is None:
        b = int(input("Enter the number of bins (integer): "))
    if q is None:
        q = int(input("Enter the number of results to be returned (integer): "))

    print_setup()

    top_q_accumulator = 0  # accumulator of the places target subgroup is in the top-q
    hit_rate_counter = 0  # counter of how many times subgroup is in the top-q
    miss_rate_counter = 0  # counter of how many times subgroup is NOT in the top-q
    start_time = time.time()
    for i in tqdm(range(STARTING_INDEX, ENDING_INDEX + 1)):
        try:
            data = pd.read_csv(PATH + DEFAULT_FILE_NAME + str(i) + ".csv", delimiter=",")
        except Exception as e:
            if i > int(STARTING_INDEX):
                print("Exception", e, " at file index", i, "\ncontinuing as its not the first file")
                continue
            else:
                print("Exception", e, " at the first file, stopping...")
                raise Exception

        dataset = data_refinement.DataSet(data, targets, descriptors)
        result = beam_search.beam_search(w, d, b, q, dataset)

        value = process_result(result)
        if value == -1:
            miss_rate_counter += 1
        else:
            hit_rate_counter += 1
            top_q_accumulator += value

        # if there is only one file to read, print the top-q
        if ENDING_INDEX - STARTING_INDEX == 0:
            print_result(result)

    end_time = time.time()
    if hit_rate_counter != 0:
        average_place = top_q_accumulator / hit_rate_counter
        print("Average place in the top-q:", average_place)
    else:
        print("Subgroup never found in top-q")
    print("Miss rate:", miss_rate_counter)
    print("\nAnalysis completed in %.0f seconds" % (end_time - start_time))
