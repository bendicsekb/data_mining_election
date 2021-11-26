# 2AMM20 Research Topic in Data Mining Group 3
import heapq
import sys
import os
import pandas as pd
import time
import tqdm

import data_refinement
import beam_search


# Synthetic dataset parameters
#PATH = "Synthetic/reversed/"  # file path to the datasets folder and should and with /
#PATH = "Synthetic/pairwise_swapped/"
PATH = "Synthetic/last_to_first/"
FILE_NR_THRESHOLD = 5

# Beam search parameters (all integers)
w = 20  # None  # beam width
d = 3  # None  # search depth
b = 1  # None  # not important for synthetic data experiment
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
        PATH = str(input("Enter the file path to the datasets: "))

    TARGET_DESCRIPTION = set_target_description()

    for (root, dirs, files) in os.walk(PATH):
        if len(dirs) == 0:
            for file in files:
                data = pd.read_csv(os.path.join(root, file), delimiter=",", index_col=0)
                #print(data.head())
                #input("Please check if the data got read in correctly\n"
                #      "if not then interrupt and change the parameters, or type any key to continue")
                break
            break

    if w is None:
        w = int(input("Enter the beam width size (integer): "))
    if d is None:
        d = int(input("Enter the beam search depth (integer): "))
    if b is None:
        b = int(input("Enter the number of bins (integer): "))
    if q is None:
        q = int(input("Enter the number of results to be returned (integer): "))

    selected_methods = [data_refinement.Method.OUR_N, data_refinement.Method.OUR_NONE, 
                        data_refinement.Method.OUR_SQRT, data_refinement.Method.OUR_ENTROPY,
                        data_refinement.Method.NORM, data_refinement.Method.LABELWISE, data_refinement.Method.PAIRWISE]
    output_files = [pd.DataFrame(columns=["Number of rows",
     "Number of descriptors", "Number of targets", 
     "Average position", "Miss rate", "Average Duration (s)"]) for _ in range(selected_methods[-1].value + 1)]
    for (root, dirs, files) in os.walk(PATH):
        if len(dirs) == 0:
            for method in selected_methods:
                folder_name = os.path.split(root)[-1]
                print("\nStarting on", folder_name, f"\t Method: {data_refinement.Method(method).name}")
                descriptors, targets = [], []

                top_q_accumulator = 0  # accumulator of the places target subgroup is in the top-q
                hit_rate_counter = 0  # counter of how many times subgroup is in the top-q
                miss_rate_counter = 0  # counter of how many times subgroup is NOT in the top-q
                start_time = time.time()
                progress_bar = tqdm.tqdm(total=min(len(files), FILE_NR_THRESHOLD))
                acc_time = file_count = 0
                
                for file in files:
                    data = pd.read_csv(os.path.join(root, file), delimiter=",", index_col=0)
                    if len(descriptors) == 0 and len(targets) == 0:
                        for column in data.columns:
                            if "descriptor" in column:
                                descriptors.append(column)
                            elif "party" in column:
                                targets.append(column)
                    dataset = data_refinement.DataSet(data, targets, descriptors)
                    b_start = time.time()
                    result = beam_search.beam_search(w, d, b, q, dataset, method)
                    b_end = time.time()
                    acc_time += (b_end - b_start)
                    file_count += 1

                    value = process_result(result)
                    if value == -1:
                        miss_rate_counter += 1
                    else:
                        hit_rate_counter += 1
                        top_q_accumulator += value

                    progress_bar.update(1)

                    if file_count > FILE_NR_THRESHOLD:
                        break

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
                output_files[method.value].loc[len(output_files[method.value])] = [nrows, ndescr, ntarget, round(average_place, 5) if average_place is not pd.NA else pd.NA, miss_rate_counter, round(acc_time / file_count, 2)]
                output_files[method.value].to_csv("/".join(root.split("/")[0:len(root.split("/"))-1]) + f"/results_{data_refinement.Method(method).name}.csv", sep=",", index=False)

                del data, dataset
