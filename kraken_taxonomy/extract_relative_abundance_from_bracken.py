import concurrent.futures
from multiprocessing import freeze_support

import pandas as pd


# define a function to process each accession
def process_accession(acc, rank='species'):
    try:
        df = pd.read_table(
            f"./data/{acc}/{rank}_relative_abundance.bracken",
            skipinitialspace=True,
        )
        df = df[["name", "fraction_total_reads"]].set_index("name", drop=True).T
        df.insert(0, "Sample", str(acc))  
        return df
    except FileNotFoundError as error:
        print(f"Report not found for {acc}", error)
        return None


if __name__ == "__main__":
    freeze_support()
    # read and process accessions concurrently
    metadata = pd.read_csv("../metadata.csv", sep=";")
    accessions = metadata[["Run"]]["Run"].to_list()

    result = pd.concat(
        [
            df
            for df in concurrent.futures.ProcessPoolExecutor().map(
                process_accession, accessions
            )
            if df is not None
        ]
    )
    result = result.fillna(0)
    result = result.set_index("Sample", drop=True)
    result.to_csv("species_relative_abundance.csv", sep=";")
    print(result.shape)
