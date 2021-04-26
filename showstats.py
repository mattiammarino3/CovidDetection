import os
import sys
import csv
import numpy as np
import pandas as pd
from nnperf.stats import nnPerf

def run(fileList):
    statList = []
    nperf_obj = nnPerf()

    for filename in fileList:
        dir = os.path.split(filename)
        name = dir[1].split(".")
        k1, v1 = nperf_obj.getPerfStatfromCSV(name[0], filename, show=True)
        statList.append(v1)

    statDF = pd.DataFrame(statList, columns=k1)
    print (statDF.head())

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print ("{} list of csv_files".format(sys.argv[0]))
        exit(1)

    run(sys.argv[1:])

