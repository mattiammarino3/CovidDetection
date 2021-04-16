import os
import sys
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class nnPerf():
    def __init__(self):
        self.cvs_filepath = ""

    def getPerfStatDF(self):
        return self.statDF
    
    """ Save epoch, accuracy, f1score to CSV file  
    input: 
        csv_file: string -> CSV file path 
        epoch: int -> epoch
        mode: str -> writing mode: default "a"
        accuracy: float -> accuracy 
        f1score: float -> f1score 
    output:
        None
    """
    def saveAccToCSV(self, csv_filepath: str = "", mode: str = "a", epoch: int=0, accuracy: float=0.0, f1score: float=0.0) -> None:
        try:
            if len(self.cvs_filepath) == 0:
                if len(csv_filepath) == 0:
                    raise Exception('no filename has provided', 1)
                else:
                    self.csv_filepath = csv_filepath

            with open(self.csv_filepath, mode=mode) as csv_file:
                csv_writerObj = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writerObj.writerow([epoch, accuracy, f1score])

        except Exception as e:
            print("error: {}".format(e.args))
        except OSError as err:
            print("error: {0}".format(err))
        except ValueError:
            print("Could not convert data to an integer.")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    """ Get performance(CPU, GPU and their memoery usage) statistics from CSV file  
    input: 
        csv_file: string -> CSV file path 
        show: bool -> show statistics data (mostly debugging purposes)
    output:
        None
    """
    def getPerfStatfromCSV(self, nnName: str = "", csv_filepath: str = "", show=True):
        retStatMap = {}
        try:
            if len(self.cvs_filepath) == 0:
                if len(csv_filepath) == 0:
                    raise Exception('no filename has provided', 1)
                else:
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
            retStatMap["CPU_TOTAL_TIME"] = cpuTotalTime

            # GPU time - gpu total
            gpuInMSDF = self.statDF.loc[self.statDF["GPU_TOTAL_UOM"] == 'ms']
            gpuInUSDF = self.statDF.loc[self.statDF["GPU_TOTAL_UOM"] == 'us']
            gpuTotalTimeMs = gpuInMSDF["GPU_TOTAL"].sum(axis = 0, skipna = True)
            gpuTotalTimeUs = gpuInUSDF["GPU_TOTAL"].sum(axis = 0, skipna = True)
            gpuTotalTime = gpuTotalTimeMs + gpuTotalTimeUs/1000

            retStatMap["GPU_TOTAL_TIME"] = gpuTotalTime

            # CPU Mem - cpu mem max
            cpuMemMBMF = self.statDF.loc[self.statDF["CPU_MEM_UOM"] == 'kb']
            minCPUMem = cpuMemMBMF["CPU_MEM"].min()
            maxCPUMem = cpuMemMBMF["CPU_MEM"].max()

            retStatMap["CPU_MEM_MIN"] = minCPUMem
            retStatMap["CPU_MEM_MAX"] = maxCPUMem

            # GPU Mem - cpu mem max
            gpuMemKBMF = self.statDF.loc[self.statDF["GPU_MEM_UOM"] == 'kb']
            if gpuMemKBMF.empty == True:
                minGPUMem = 0
                maxGPUMem = 0
            else:
                minGPUMem = gpuMemKBMF["GPU_MEM"].min(axis = 0, skipna = True)
                maxGPUMem = gpuMemKBMF["GPU_MEM"].max(axis = 0, skipna = True)
            
            retStatMap["GPU_MEM_MIN"] = minCPUMem
            retStatMap["GPU_MEM_MAX"] = maxCPUMem

            totalNOC = self.statDF["NUMBER_OF_CALLS"].sum(axis = 0)
            retStatMap["TOTAL_NOC"] = totalNOC

            if show:
                print ("---------------------------------------------------------------------------------")
                print ("cpu_total_time = {:.2f} ms, gpu_total_time = {} ms".format(cpuTotalTime, gpuTotalTime))
                print ("cpu_mem_max = {} kb, gpu_mem_max = {} kb".format(maxCPUMem, maxGPUMem))
                print ("total_number_of_call = {} times".format(totalNOC))
                print ("avg_cpu_time_per_module = {:.2f} ms".format(cpuTotalTime/totalNOC))
                print ("avg_gpu_time_per_module = {:.2f} ms".format(gpuTotalTime/totalNOC))

            retDic =  {"cpu_total_time": cpuTotalTime, "gpu_total_time": gpuTotalTime, "total_number_of_call": totalNOC,\
                       "avg_cpu_time_per_module": (cpuTotalTime/totalNOC), "avg_gpu_time_per_module": (gpuTotalTime/totalNOC)}
        except Exception as e:
            print("error: {}".format(e.args))
        except OSError as err:
            print("error: {0}".format(err))
        except ValueError:
            print("Could not convert data to an integer.")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        return retStatMap

    def showPerfStatGraphs(self, dataFrame):
        return 0

# test purpose only
nperf_obj = nnPerf()
retData = nperf_obj.getPerfStatfromCSV("ResNet", "ResNetKPI.csv", show=True)
print (retData)
statDF = pd.DataFrame.from_dict(retData)
print (statDF.head())

retData = nperf_obj.getPerfStatfromCSV("DenseNet", "DenseNetKPI.csv", show=True)
statDF.concat(pd.DataFrame.from_dict(retData))

print (statDF.head())
#nperf_obj.saveAccToCSV("test.csv","a", 0, 0.99, 0.99)