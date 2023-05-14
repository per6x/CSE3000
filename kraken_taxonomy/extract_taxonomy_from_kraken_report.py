import concurrent.futures
import pandas as pd
from multiprocessing import freeze_support


# define a function to process each accession
def process_accession(acc):
    try:
        df = pd.read_table(
            f"./data/{acc}/report.txt",
            names=["%", "total count", "self count", "taxonomy", "tax id", "name"],
            skipinitialspace=True,
        )
        d_index = list(
            filter(
                lambda x: x > df.index[df["name"] == "Bacteria"].to_list()[0],
                df.index[df["taxonomy"] == "D"],
            )
        )
        if len(d_index) == 0:
            df = df.loc[::, ::]
        else:
            df = df.loc[: d_index[0] - 1, ::]
        df = df.loc[(df["taxonomy"] == "S")]
        df = df[["total count", "name"]].set_index("name").T
        df.insert(0, "Sample", str(acc))
        if df.shape[1] > 2000:
            return df
        else:
            print(f"BAD accession: {acc}")
            return None
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
    result.to_csv("features.csv", sep=";")
