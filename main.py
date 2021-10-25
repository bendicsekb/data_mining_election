# 2AMM20 Research Topic in Data Mining Group 3
import heapq
import sys
import os
import pandas as pd
import time
import tqdm

import data_refinement
import beam_search

# party votes columns
parties_abs = ['VVD', 'D66', 'PVV (Partij voor de Vrijheid)', 'CDA', 'SP (Socialistische Partij)', 'Partij van de Arbeid (P.v.d.A.)', 'GROENLINKS', 'Forum voor Democratie', 'Partij voor de Dieren', 'ChristenUnie', 'Volt', 'JA21', 'Staatkundig Gereformeerde Partij (SGP)', 'DENK', '50PLUS', 'BBB', 'BIJ1', 'CODE ORANJE', 'NIDA', 'Splinter', 'Piratenpartij', 'JONG', 'Trots op Nederland (TROTS)', 'Lijst Henk Krol', 'NLBeter', 'Blanco (Zeven, A.J.L.B.)', 'LP (Libertaire Partij)', 'OPRECHT', 'JEZUS LEEFT', 'DE FEESTPARTIJ (DFP)', 'U-Buntu Connected Front', 'Vrij en Sociaal Nederland', 'Partij van de Eenheid', 'Wij zijn Nederland', 'Partij voor de Republiek', 'Modern Nederland', 'De Groenen']  # column names of the target attributes
parties_rel = ['VVD (%)', 'D66 (%)', 'PVV (Partij voor de Vrijheid) (%)', 'CDA (%)', 'SP (Socialistische Partij) (%)', 'Partij van de Arbeid (P.v.d.A.) (%)', 'GROENLINKS (%)', 'Forum voor Democratie (%)', 'Partij voor de Dieren (%)', 'ChristenUnie (%)', 'Volt (%)', 'JA21 (%)', 'Staatkundig Gereformeerde Partij (SGP) (%)', 'DENK (%)', '50PLUS (%)', 'BBB (%)', 'BIJ1 (%)', 'CODE ORANJE (%)', 'NIDA (%)', 'Splinter (%)', 'Piratenpartij (%)', 'JONG (%)', 'Trots op Nederland (TROTS) (%)', 'Lijst Henk Krol (%)', 'NLBeter (%)', 'Blanco (Zeven, A.J.L.B.) (%)', 'LP (Libertaire Partij) (%)', 'OPRECHT (%)', 'JEZUS LEEFT (%)', 'DE FEESTPARTIJ (DFP) (%)', 'U-Buntu Connected Front (%)', 'Vrij en Sociaal Nederland (%)', 'Partij van de Eenheid (%)', 'Wij zijn Nederland (%)', 'Partij voor de Republiek (%)', 'Modern Nederland (%)', 'De Groenen (%)']  # column names of the target attributes

# Synthetic dataset parameters
PATH = "C:/Users/Studio/Documents/projects/TUe/last_to_first"  # file path to the datasets folder

# Target, descriptors, or unwanted descriptor definition
# If descriptors are empty, all attributes in the dataset, not in targets and not in unwanted_descriptors will be used
# targets = []
# descriptors = []
# unwanted_descriptors = []
# list(set().union(['RegioNaam', 'Region code', 'Kiesgerechtigden', 'Opkomst', 'OngeldigeStemmen', 'BlancoStemmen', 'GeldigeStemmen'], parties_abs))

# Beam search parameters (all integers)
w = 20  # None  # beam width
d = 3  # None  # search depth
b = 1  # None  # static binning bin size
q = 10  # None  # top q subgroups to return


# def print_result(result: list[(float, int, data_refinement.Description)]):
def print_result(result):
    print("\nFound subgroups")
    for i in reversed(range(len(result))):
        sys.stdout.write(str(i+1))
        sys.stdout.write(": ")
        heapq.heappop(result)[2].print_description()


# Set target description
def set_target_description():
    target_description = data_refinement.Description()
    rule = data_refinement.Rule(data_refinement.RuleType.BINARY, "descriptor1", "==", 1.0)
    target_description.add_rule(rule)
    rule = data_refinement.Rule(data_refinement.RuleType.BINARY, "descriptor2", "==", 1.0)
    target_description.add_rule(rule)
    return target_description


# def process_result(top_q: list[(float, int, data_refinement.Description)]):
def process_result(top_q):
    if len(top_q) == 0:  # although it should not happen
        return -1

    top_q_sorted_list = sorted(top_q.copy(), key=lambda tup: tup[0], reverse=True)  # get top_q, sort descendlingly on quality
    description_list = [tup[2] for tup in top_q_sorted_list]  # Keep only the descriptions

    # Check for every description
    for desc in description_list:
        count = 0  # counter to keep track of the number of rule matches
        if len(desc.rules) == len(TARGET_DESCRIPTION.rules):  # only match if they have the same number of rules
            for desc_rule in desc.rules:
                for target_rule in TARGET_DESCRIPTION.rules:
                    # compare each rule in the top-q description against our target descriptions rule
                    if desc_rule.to_string() == target_rule.to_string():
                        # if they are equivalent, count 1 and break out of the current loop
                        count += 1
                        break

            if count == len(TARGET_DESCRIPTION.rules):
                return description_list.index(desc) + 1

    return -1


if __name__ == '__main__':
    if PATH == "" or PATH is None:
        #PATH = str(input("Enter the file path to the datasets: "))
        raise(Exception("No PATH given"))

    TARGET_DESCRIPTION = set_target_description()

    for (root, dirs, files) in os.walk(PATH):
        if len(dirs) == 0:
            for file in files:
                if "nrows1000" not in file:
                    data = pd.read_csv(os.path.join(root, file), delimiter=",", index_col=0)
                    print(data.head())
                    # input("Please check if the data got read in correctly\n"
                    #       "if not then interrupt and change the parameters, or type any key to continue")
                    print("Please check if the data got read in correctly\n"
                        "if not then interrupt and change the parameters, or type any key to continue")
                    break
            break

    if w is None:
        # w = int(input("Enter the beam width size (integer): "))
        raise(Exception("No beam width specified"))
    if d is None:
        # d = int(input("Enter the beam search depth (integer): "))
        raise(Exception("No beam search depth specified"))
    if b is None:
        # b = int(input("Enter the number of bins (integer): "))
        raise(Exception("# of bins not specified"))
    if q is None:
        # q = int(input("Enter the number of results to be returned (integer): "))
        raise(Exception("# of results to be returned not specified"))

    output_file = pd.DataFrame(columns=["Number of rows", "Number of descriptors", "Number of targets", "Average position", "Miss rate"])
    for (root, dirs, files) in os.walk(PATH):
        if len(dirs) == 0 and "nrow500" in root:
            folder_name = os.path.split(root)[-1]
            print("\nStarting on", folder_name)
            descriptors, targets = [], []

            top_q_accumulator = 0  # accumulator of the places target subgroup is in the top-q
            hit_rate_counter = 0  # counter of how many times subgroup is in the top-q
            miss_rate_counter = 0  # counter of how many times subgroup is NOT in the top-q
            start_time = time.time()
            progress_bar = tqdm.tqdm(total=len(files))
            for file in files:
                data = pd.read_csv(os.path.join(root, file), delimiter=",", index_col=0)
                if len(descriptors) == 0 and len(targets) == 0:
                    for column in data.columns:
                        if "descriptor" in column:
                            descriptors.append(column)
                        elif "party" in column:
                            targets.append(column)
                dataset = data_refinement.DataSet(data, targets, descriptors)
                result = beam_search.beam_search(w, d, b, q, dataset)

                value = process_result(result)
                if value == -1:
                    miss_rate_counter += 1
                else:
                    hit_rate_counter += 1
                    top_q_accumulator += value

                progress_bar.update(1)

            progress_bar.close()
            end_time = time.time()
            if hit_rate_counter != 0:
                average_place = top_q_accumulator / hit_rate_counter
            else:
                average_place = pd.NA

            print("Analysis of", folder_name, " completed in %.0f seconds" % (end_time - start_time))
            print("Average place:", average_place, " - miss rate:", miss_rate_counter)

            nrows = int(folder_name.split("_")[0][4:])
            ndescr = int(folder_name.split("_")[1][6:])
            ntarget = int(folder_name.split("_")[2][7:])
            output_file.loc[len(output_file)] = [nrows, ndescr, ntarget, round(average_place, 5), miss_rate_counter]
            output_file.to_csv("/".join(root.split("/")[0:len(root.split("/"))-1]) + "/results.csv", sep=",", index=False)

            del data, dataset
