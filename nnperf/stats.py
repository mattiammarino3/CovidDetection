import os
import sys
import numpy as np
import pandas as pd

class nnPerf():
    def __init__(self):
        self.cvs_filepath = None

    def getStatDF(self):
        return self.statDF

    def getStatfromCSV(self, csv_filepath, show=True):
        try:
            self.csv_filepath = csv_filepath

            self.statDF = pd.read_csv(csv_filepath, header=0) #, skiprows=[0])

            if show:
                print (self.statDF.head(5))

            # CPU time - cpu total
            cpuInMSDF = self.statDF.loc[self.statDF["CPU_TOTAL_UOM"] == 'ms']
            cpuInUSDF = self.statDF.loc[self.statDF["CPU_TOTAL_UOM"] == 'us']
            cpuTotalTimeMs = cpuInMSDF["CPU_TOTAL"].sum(axis = 0, skipna = True)
            cpuTotalTimeUs = cpuInUSDF["CPU_TOTAL"].sum(axis = 0, skipna = True)
            cpuTotalTime = cpuTotalTimeMs + cpuTotalTimeUs/1000

            # GPU time - gpu total
            gpuInMSDF = self.statDF.loc[self.statDF["GPU_TOTAL_UOM"] == 'ms']
            gpuInUSDF = self.statDF.loc[self.statDF["GPU_TOTAL_UOM"] == 'us']
            gpuTotalTimeMs = gpuInMSDF["GPU_TOTAL"].sum(axis = 0, skipna = True)
            gpuTotalTimeUs = gpuInUSDF["GPU_TOTAL"].sum(axis = 0, skipna = True)
            gpuTotalTime = gpuTotalTimeMs + gpuTotalTimeUs/1000

            # CPU Mem - cpu mem max
            cpuMemMBMF = self.statDF.loc[self.statDF["CPU_MEM_UOM"] == 'kb']
            maxCPUMem = cpuMemMBMF["CPU_MEM"].max()

            # GPU Mem - cpu mem max
            gpuMemKBMF = self.statDF.loc[self.statDF["GPU_MEM_UOM"] == 'kb']
            if gpuMemKBMF.empty == True:
                maxGPUMem = 0
            else:
                maxGPUMem = gpuMemKBMF["GPU_MEM"].max(axis = 0, skipna = True)

            totalNOC = self.statDF["NUMBER_OF_CALLS"].sum(axis = 0)

            if show:
                print ("---------------------------------------------------------------------------------")
                print ("cpu_total_time = {:.2f} ms, gpu_total_time = {} ms".format(cpuTotalTime, gpuTotalTime))
                print ("cpu_mem_max = {} kb, gpu_mem_max = {} kb".format(maxCPUMem, maxGPUMem))
                print ("total_number_of_call = {} times".format(totalNOC))
                print ("avg_cpu_time_per_module = {:.2f} ms".format(cpuTotalTime/totalNOC))
                print ("avg_gpu_time_per_module = {:.2f} ms".format(gpuTotalTime/totalNOC))

            retDic =  {"cpu_total_time": cpuTotalTime, "gpu_total_time": gpuTotalTime, "total_number_of_call": totalNOC,\
                       "avg_cpu_time_per_module": (cpuTotalTime/totalNOC), "avg_gpu_time_per_module": (gpuTotalTime/totalNOC)}
        except OSError as err:
            print("error: {0}".format(err))
        except ValueError:
            print("Could not convert data to an integer.")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

nperf_obj = nnPerf()
nperf_obj.getStatfromCSV("ResNetKPI.csv", show=True)





