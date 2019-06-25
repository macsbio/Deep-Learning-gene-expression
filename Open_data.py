import pickle
import numpy as np
import Types

#this file is useful for investigating pickle files (e.g. data_X20_1.p)

filename2 = "/data_1.p"
filename3 = "/data_2.p"

infile = open(filename2,'rb')
expr,compounds,genes,variances = pickle.load(infile)
infile.close()
print(genes)
print("\n")

infile = open(filename3,'rb')
expr,compounds,genes,variances = pickle.load(infile)
infile.close()
print(genes)

