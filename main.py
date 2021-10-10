# 2AMM20 Research Topic in Data Mining Group 3
import heapq
import sys
import pandas as pd

import data_refinement
import beam_search

# party votes columns
parties_abs = ['VVD', 'D66', 'PVV (Partij voor de Vrijheid)', 'CDA', 'SP (Socialistische Partij)', 'Partij van de Arbeid (P.v.d.A.)', 'GROENLINKS', 'Forum voor Democratie', 'Partij voor de Dieren', 'ChristenUnie', 'Volt', 'JA21', 'Staatkundig Gereformeerde Partij (SGP)', 'DENK', '50PLUS', 'BBB', 'BIJ1', 'CODE ORANJE', 'NIDA', 'Splinter', 'Piratenpartij', 'JONG', 'Trots op Nederland (TROTS)', 'Lijst Henk Krol', 'NLBeter', 'Blanco (Zeven, A.J.L.B.)', 'LP (Libertaire Partij)', 'OPRECHT', 'JEZUS LEEFT', 'DE FEESTPARTIJ (DFP)', 'U-Buntu Connected Front', 'Vrij en Sociaal Nederland', 'Partij van de Eenheid', 'Wij zijn Nederland', 'Partij voor de Republiek', 'Modern Nederland', 'De Groenen']  # column names of the target attributes
parties_rel = ['VVD (%)', 'D66 (%)', 'PVV (Partij voor de Vrijheid) (%)', 'CDA (%)', 'SP (Socialistische Partij) (%)', 'Partij van de Arbeid (P.v.d.A.) (%)', 'GROENLINKS (%)', 'Forum voor Democratie (%)', 'Partij voor de Dieren (%)', 'ChristenUnie (%)', 'Volt (%)', 'JA21 (%)', 'Staatkundig Gereformeerde Partij (SGP) (%)', 'DENK (%)', '50PLUS (%)', 'BBB (%)', 'BIJ1 (%)', 'CODE ORANJE (%)', 'NIDA (%)', 'Splinter (%)', 'Piratenpartij (%)', 'JONG (%)', 'Trots op Nederland (TROTS) (%)', 'Lijst Henk Krol (%)', 'NLBeter (%)', 'Blanco (Zeven, A.J.L.B.) (%)', 'LP (Libertaire Partij) (%)', 'OPRECHT (%)', 'JEZUS LEEFT (%)', 'DE FEESTPARTIJ (DFP) (%)', 'U-Buntu Connected Front (%)', 'Vrij en Sociaal Nederland (%)', 'Partij van de Eenheid (%)', 'Wij zijn Nederland (%)', 'Partij voor de Republiek (%)', 'Modern Nederland (%)', 'De Groenen (%)']  # column names of the target attributes

# Dataset parameters
PATH = "datasets/Demographic_and_election_dataset.csv"  # file path to the dataset
targets = parties_rel
descriptors = []  # column names of the descriptor attributes
unwanted_descriptors = list(set().union(['RegioNaam', 'Region code', 'Kiesgerechtigden', 'Opkomst', 'OngeldigeStemmen', 'BlancoStemmen', 'GeldigeStemmen'], parties_abs))

# 'RegioNaam', 'Region code', 'Total population (nr.)', 'Men (%)', 'Women (%)', '<5 years old (%)', '5-9 years old (%)', '10-14 years old (%)', '15-19 years old (%)', '20-24 years old (%)', '25-44 years old (%)', '45-64 years old (%)', '65-79 years old (%)', '80+ years old  (%)', 'Total demographic pressure (%)', 'Green pressure (%)', 'Grey pressure (%)', 'Unmarried (%)', 'Married (%)', 'Divorced (%)', 'Widowed (%)', 'Dutch background (%)', 'Migration background - any (%)', 'Migration background - western (%)', 'Migration background - any non-western (%)', 'Migration background - Maroccan (%)', 'Migration background - former Dutch Antilles, Aruba  (%)', 'Migration background - Suriname (%)', 'Migration background - Turkey (%)', 'Migration background - remaining non-western (%)', 'Population density (people/km2)', 'Single person households (%)', 'Households without children (%)', 'Households with children (%)', 'Average household size (people/household)', 'Total housing stock (nr.)', 'Newly constructed houses (%)', 'Housing density (houses/km2)', 'Owner-occupied houses (%)', 'Rental houses (%)', 'House ownership unknown (%)', 'Average house price (x 1000EUR)', 'Total students (nr.)', 'Students - secondary education (%)', 'Students - bol (%)', 'Students - bbl (%)', 'Students - hbo (%)', 'Students - university (%)', 'Companies (nr.)', 'Companies by type - agriculture, forestry and fishery (%)', 'Companies by type - industry and engery (%)', 'Companies by type - trade and catering industry (%)', 'Companies by type - transport, information and comunication (%)', 'Companies by type - financial services and real-estate (%)', 'Companies by type - business services (%)', 'Companies by type - culture, recreation and other (%)', 'Cattle (nr.)', 'Sheep (nr.)', 'Goats (nr.)', 'Horses (nr.)', 'Pigs (nr.)', 'Chickens (nr.)', 'Turkeys (nr.)', 'Ducks for slaughter (nr.)', 'Other poultry (nr.)', 'Rabbits (nr.)', 'Fur animals (nr.)', 'Cultivated land (are)', 'Cultivated land by type - agriculture (%)', 'Cultivated land by type - horticulture, open ground (%)', 'Cultivated land by type - horticulture, under glass (%)', 'Cultivated land by type - permanent grassland (%)', 'Cultivated land by type - natural grassland (%)', 'Cultivated land by type - temporary grassland (%)', 'Cultivated land by type - green fodder crops (%)', 'Cars (per 1000 inhabitants)', 'Privately owned cars (per 1000 inhabitants)', 'Motorcycles (per 1000 inhabitants)', 'Mopeds (per 1000 inhabitants)', 'Total road length (km)', 'Road owned by municipality (%)', 'Road owned by province (%)', 'Road owned by state (%)', 'Total area (km2)', 'Districts (nr.)', 'Neighbourhoods (nr.)'

# Beam search parameters (all integers)
w = 10  # None  # beam width
d = 3  # None  # search depth
b = 5  # None  # static binning bin size
q = 100  # None  # top q subgroups to return


def print_setup():
    print("\nRunning top-q beam search on dataset", PATH)
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
    data = pd.read_csv(PATH, delimiter=",")

    print("Please check if the data got read in correctly, if not change the read_csv parameters")
    print(data.head())

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
    dataset = data_refinement.DataSet(data, targets, descriptors)
    result = beam_search.beam_search(w, d, b, q, dataset)
    print_result(result)



