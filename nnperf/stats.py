import os
import numpy as np
import pandas as pd

class nnPerf():
    def __init__(self):
        self.cvs_filepath = None
    
    def getStatDF(self):
        return self.statDF

    def getStatfromCSV(self, csv_filepath, header=None):
        try: 
            self.csv_filepath = csv_filepath
            self.statDF = pd.read_csv(csv_filepath, header=header)

            # CPU time - cpu total
            cpuInMSDF = self.statDF.loc[self.statDF[4] == 'ms']
            cpuInUSDF = self.statDF.loc[self.statDF[4] == 'us']
            cpuTotalTimeMs = cpuInMSDF[3].sum(axis = 0, skipna = True)
            cpuTotalTimeUs = cpuInUSDF[3].sum(axis = 0, skipna = True)
            cpuTotalTime = cpuTotalTimeMs + cpuTotalTimeUs/1000

            # GPU time - gpu total
            gpuInMSDF = self.statDF.loc[self.statDF[6] == 'ms']
            gpuInUSDF = self.statDF.loc[self.statDF[6] == 'us']
            gpuTotalTimeMs = gpuInMSDF[5].sum(axis = 0, skipna = True)
            gpuTotalTimeUs = gpuInUSDF[5].sum(axis = 0, skipna = True)
            gpuTotalTime = gpuTotalTimeMs + gpuTotalTimeUs/1000

            # CPU Mem - cpu mem max
            cpuMemMBMF = self.statDF.loc[self.statDF[10] == 'Mb']
            maxCPUMem = cpuMemMBMF[9].max()

            # GPU Mem - cpu mem max
            gpuMemMBMF = self.statDF.loc[self.statDF[14] == 'Mb']
            if gpuMemMBMF.empty == True:
                maxGPUMem = 0
            else:
                maxGPUMem = gpuMemMBMF[13].max(axis = 0, skipna = True)

            totalNOC = self.statDF[15].sum(axis = 0)

            print (self.statDF.head(5), header)
            print ("---------------------------------------------------------------------------------")
            print ("cpu_total_time = {:.2f}, gpu_total_time = {}".format(cpuTotalTime, gpuTotalTime))
            print ("cpu_mem_max = {}, gpu_mem_max = {}".format(maxCPUMem, maxGPUMem))
            print ("total_number_of_call = {}".format(totalNOC))
            print ("avg_cpu_time_per_module = {:.2f}".format(cpuTotalTime/totalNOC))
            print ("avg_gpu_time_per_module = {:.2f}".format(gpuTotalTime/totalNOC))

            retDic =  {"cpu_total_time": cpuTotalTime, "gpu_total_time": gpuTotalTime, "total_number_of_call": totalNOC,\
                       "avg_cpu_time_per_module": (cpuTotalTime/totalNOC), "avg_gpu_time_per_module": (gpuTotalTime/totalNOC)}

        except Exception as e:
            print (e)

nperf_obj = nnPerf()
nperf_obj.getStatfromCSV("sample_data.csv")
        

        
            

