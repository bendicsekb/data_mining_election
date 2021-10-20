import heapq
import pandas as pd

import data_refinement
import quality_measure

LOGGING_PATH = None  # "logging/logging.csv"


def beam_search(width: int, depth: int, bins: int, q: int, data: data_refinement.DataSet):
    candidate_queue = list()
    result_set = []
    unique_counter = int(2e32)  # really large number

    # dataframe containing useful information on each description's quality, to be later outputted as csv
    # Note that changes to for example the columns, will require changes in the quality_measure.py functions
    quality_measure_data = pd.DataFrame(data=None, columns=["Description", "Quality", "Size of subgroup", "Size of complement", "Numerator", "Denominator"])

    empty_description = data_refinement.Description()
    heapq.heappush(candidate_queue, (0, 0, empty_description))

    for i in range(depth):
        #time_start = time.time()
        beam = []

        while len(candidate_queue) != 0:
            seed = candidate_queue.pop(0)[2]  # pop the (quality, description) tuple and keep only the description
            descriptions = data_refinement.refine(seed, data, bins)
            for desc in descriptions:
                unique_counter -= 1

                quality_data = quality_measure.set_quality(desc, data, 0, quality_measure_data)
                if quality_data is not None:
                    quality_measure_data = quality_data

                # if the result set size is not yet q, simply push the description
                # Note that the tuples consist of (quality, random number, description), because whenever there are two
                # equivalent qualities, heapq will try to compare the description which it cannot and hence errors.
                # The unique counter fixes this on the short term
                if len(result_set) < q:
                    heapq.heappush(result_set, (desc.quality, unique_counter, desc))
                else:
                    heapq.heappushpop(result_set, (desc.quality, unique_counter, desc))

                # do the same for the beam, i.e. keep only descriptions with high qualities in the beam
                if len(beam) < width:
                    heapq.heappush(beam, (desc.quality, unique_counter, desc))
                else:
                    heapq.heappushpop(beam, (desc.quality, unique_counter, desc))

        # candidate queue is empty, so remove all descriptions from the beam
        # and insert them into the candidate queue for the next iteration
        while len(beam) != 0:
            candidate_queue.append(beam.pop(0))

        #time_end = time.time()
        #print("Depth", i, "completed in %.0f seconds" % (time_end - time_start))

    if LOGGING_PATH is not None or LOGGING_PATH != "":
        quality_measure_data.to_csv(LOGGING_PATH, sep=";", index=False)

    return result_set
