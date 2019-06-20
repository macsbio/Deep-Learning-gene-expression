import pickle
import numpy as np
import Types

# Dan: this file is useful for investigating pickle files (e.g. data_X20_1.p)

filename2 = "/Users/Daniel/Desktop/data_X65_1_nest.p"
filename3 = "/Users/Daniel/Desktop/data_X79_1_nest.p"

infile = open(filename2,'rb')
expr,compounds,genes,variances = pickle.load(infile)
infile.close()
print(genes)
print("\n")

infile = open(filename3,'rb')
expr,compounds,genes,variances = pickle.load(infile)
infile.close()
print(genes)

