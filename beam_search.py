import heapq

import data_refinement
import quality_measure


def beam_search(width: int, depth: int, bins: int, q: int, data: data_refinement.DataSet,
                method: data_refinement.Method):
    candidate_queue = list()
    result_set = []
    unique_counter = int(2e32)  # really large number
    empty_description = data_refinement.Description()
    heapq.heappush(candidate_queue, (0, 0, empty_description))

    for i in range(depth):
        beam = []

        while len(candidate_queue) != 0:
            seed = candidate_queue.pop(0)[2]  # pop the (quality, description) tuple and keep only the description
            descriptions = data_refinement.refine(seed, data, bins)
            for desc in descriptions:
                unique_counter -= 1

                quality_measure.set_quality(desc, data, method)

                unique = True
                for result_tup in result_set:
                    if data_refinement.check_unique_rule(desc, result_tup[2]):
                        unique = False
                        break
                for beam_tup in beam:
                    if not unique and data_refinement.check_unique_rule(desc, beam_tup[2]):
                        unique = False
                        break

                if not unique:
                    continue

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

    return result_set
