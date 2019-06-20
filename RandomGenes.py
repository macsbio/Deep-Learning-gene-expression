import pickle
import numpy as np
import scipy.io as sio
import CompoundLists
import tensorflow as tf
from PredictEncoding import leave_one_out_evaluation
from SmallParser import SmallDatasetParser, get_doubling_xy, get_x, read_data, read_nested_orths
from Processing import correct_slopes, decorrect_slopes, normalize_features, normalize_with_norms, \
    denormalize_with_norms, normalize_total, split_train_test

'''
Warning: this is VERY CPU/disk intensive and may slow down / crash your computer. 

The final output of the program will contain accuracy info, and separate files will be created for the random gene sets (x and y)
and the LOO prediction results. The genes will be printed before starting the LOO procedure.

A single execution will likely take several hours.
'''

def main():
    tf.logging.set_verbosity(tf.logging.ERROR)

    compound_list = CompoundLists.UNGENERAL_45  # Dan: I changed this line
    
    # Change these lines to change prediction direction. 
    # NOTE: currently only supports rat vitro -> vitro and rat vitro -> rat vivo
    # i.e. x_type should always be rat_vitro
    x_type = "rat_vitro"
    y_type = "human_vitro"  # Dan: change here if necessary!

    x_timepoints = 3
    y_timepoints = 3
    if y_type == 'rat_vivo':
        y_timepoints = 4     

    x_satisfied = False
    y_satisfied = False

    # Only use these lines if you already have the data files. Selection process will then be disabled.
    #x_satisfied = True
    #og_X, data_compounds, gene_list_x, X_gene_variance = pickle.load(open('Data/RatInVitro/20/data_X20_1.p', 'rb'))
    #y_satisfied = True
    #og_Y, data_compounds, gene_list_y, Y_gene_variance = pickle.load(open('Data/HumanInVitro/20/data_20_1_human.p', 'rb'))


    # Variances from ST gene list (steatosis, 50 genes)
    target_x_var = 0.001386  # rat in vitro
    target_y_var = 0.001325  # rat in vivo
    if y_type == 'human_vitro':
        target_y_var = 0.000986
    
    deviation = 0.03  # variance should be at most 3% more than the target and at least 3% less than the target
    x_var_low = target_x_var - target_x_var * deviation
    x_var_up = target_x_var + target_x_var * deviation
    y_var_low = target_y_var - target_y_var * deviation
    y_var_up = target_y_var + target_y_var * deviation
   

    print("X var range: {} - {}".format(x_var_low, x_var_up))
    print("Y var range: {} - {}".format(y_var_low, y_var_up))

    
    nested = True  #if true, input file needs to be selected below (scroll down)
    orthologs = False  #change here as desired
    
    # names of output files need to be manually adjusted (as desired)
    # to create the 'core' of a nest (e.g. 20), nested must be set to False!

    if nested == False:
        for k in range(80,81):  # desired number of genes
            numb_genes = k
            for j in range(99,100):  # desired number (i.e. names) of sets
                x_satisfied = False  # set to True if you want to select only one domain (not recommended)
                y_satisfied = False
                file1 = "data_X%d"%(numb_genes)+"_%d"%(j) + ".p"
                file2 = "data_%d"%(numb_genes)+"_%d"%(j) + "_human.p"  # change to desired domain here!

                while not x_satisfied or not y_satisfied:
                    og_X, og_Y, data_compounds, genes_x, genes_y = read_data(compound_list.copy(), x_type=x_type, y_type=y_type, \
                    gene_list='random', dataset="big", numb_genes=numb_genes, domain="both", orthologs=orthologs)
                    if not x_satisfied:
                        X, _ = normalize_total(og_X)
                        X_gene_means = np.zeros(numb_genes)
                        X_gene_variance = np.zeros(numb_genes)
                        for i in range(numb_genes):
                            X_gene_means[i] = np.mean(X[:, i * x_timepoints : i * x_timepoints + x_timepoints])
                            X_gene_variance[i] = np.var(X[:, i * x_timepoints : i * x_timepoints + x_timepoints])

                        if X_gene_variance.mean() >= x_var_low and X_gene_variance.mean() <= x_var_up:
                            print("X satisfied!")
                            x_satisfied = True
                            gene_list_x = genes_x
                            if not orthologs:
                                with open(file1, 'wb') as f:
                                    pickle.dump([og_X, data_compounds, gene_list_x, X_gene_variance], f)
                                    print("Dumped file ", file1)
                                    print("X genes: ", gene_list_x)
                        elif orthologs:  # occurs only if we select unnested orthologs (e.g. 20)
                            continue

                    if not y_satisfied:
                        Y, _ = normalize_total(og_Y)
                        Y_gene_means = np.zeros(numb_genes)
                        Y_gene_variance = np.zeros(numb_genes)
                        for i in range(numb_genes):
                            Y_gene_means[i] = np.mean(Y[:, i * y_timepoints : i * y_timepoints + y_timepoints])
                            Y_gene_variance[i] = np.var(Y[:, i * y_timepoints : i * y_timepoints + y_timepoints])

                        print("Y Mean:", Y_gene_means.mean())
                        print("Y Variance:", Y_gene_variance.mean())

                        if Y_gene_variance.mean() >= y_var_low and Y_gene_variance.mean() <= y_var_up:
                            print("Y satisfied!")
                            y_satisfied = True
                            gene_list_y = genes_y
                            if not orthologs:
                                with open(file2, 'wb') as f:
                                    pickle.dump([og_Y, data_compounds, gene_list_y, Y_gene_variance], f)
                                    print("Dumped file ", file1)
                                    print("Y genes: ", gene_list_y)
                        elif orthologs:
                            x_satisfied = False  # starting all over again

                    if x_satisfied and y_satisfied and orthologs:
                        with open(file1, 'wb') as f:
                            pickle.dump([og_X, data_compounds, gene_list_x, X_gene_variance], f)
                        with open(file2, 'wb') as f:
                            pickle.dump([og_Y, data_compounds, gene_list_y, Y_gene_variance], f)

    else:  # nested case
        numb_genes = 15  # number of genes to add
        numb_genes_core = 35  # size of set to build upon
        numb_genes_out = numb_genes + numb_genes_core
        for j in range(1,2):  #  desired number (and names) of output sets
            y_satisfied = False  # set True if only one domain is desired (not recommended)
            x_satisfied = False
            file_out1 = "data_X%d"%(numb_genes_out) + "_%d"%(j) + "_nest.p"
            file_out2 = "data_%d"%(numb_genes_out) + "_%d"%(j) + "_human_nest.p"  # Dan: adjust here if necessary
            #file_in1 = "data_X%d"%(numb_genes_core) + "_%d"%(j) + "_nest.p"
            #file_in2 = "data_%d"%(numb_genes_core) + "_%d"%(j) + "_human_nest.p"        
            file_in1 = "Data/RatInVitro/%d"%(numb_genes_core) + "/Nested/Random%d"%(numb_genes_core) + \
            "/data_X%d"%(numb_genes_core) + "_%d"%(j) + "_nest.p"
            file_in2 = "Data/HumanInVitro/%d"%(numb_genes_core) + "/Nested/Random%d"%(numb_genes_core) + \
            "/data_%d"%(numb_genes_core) + "_%d"%(j) + "_human_nest.p"  # change name here as desired

            X_core, _, gene_list_x_core, variance_x_core = pickle.load(open(file_in1, "rb"))
            Y_core, _, gene_list_y_core, variance_y_core = pickle.load(open(file_in2, "rb"))

            while not x_satisfied or not y_satisfied:
                og_X, og_Y, data_compounds, gene_list_x, gene_list_y = read_data(compound_list.copy(), y_type=y_type, \
                gene_list='random', domain="both", numb_genes=numb_genes, orthologs=orthologs, genes_provided=gene_list_x_core)
                if not x_satisfied:
                    X_temp1, _ = normalize_total(og_X)
                    X_temp2, _ = normalize_total(X_core)
                    X = np.concatenate((X_temp1, X_temp2), axis=1)
                    X_gene_means = np.zeros(numb_genes_out)
                    X_gene_variance = np.zeros(numb_genes_out)
                    for i in range(numb_genes):
                        X_gene_means[i] = np.mean(X[:, i * x_timepoints : i * x_timepoints + x_timepoints])
                        X_gene_variance[i] = np.var(X[:, i * x_timepoints : i * x_timepoints + x_timepoints])
                        if X_gene_variance[i] >= x_var_low and X_gene_variance[i] <= x_var_up:
                            print("Gene ", gene_list_x[i], " has variance ", X_gene_variance[i])

                    print("X Mean:", X_gene_means.mean())
                    print("X Variance:", X_gene_variance.mean())

                    if X_gene_variance.mean() >= x_var_low and X_gene_variance.mean() <= x_var_up:
                        print("X satisfied!")
                        x_satisfied = True
                    elif orthologs:
                        continue

                if not y_satisfied:
                    Y_temp1, _ = normalize_total(og_Y)
                    Y_temp2, _ = normalize_total(Y_core)
                    Y = np.concatenate((Y_temp1, Y_temp2), axis=1)
                    Y_gene_means = np.zeros(numb_genes_out)
                    Y_gene_variance = np.zeros(numb_genes_out)
                    for i in range(numb_genes_out):
                        Y_gene_means[i] = np.mean(Y[:, i * y_timepoints : i * y_timepoints + y_timepoints])
                        Y_gene_variance[i] = np.var(Y[:, i * y_timepoints : i * y_timepoints + y_timepoints])

                    print("Y Mean:", Y_gene_means.mean())
                    print("Y Variance:", Y_gene_variance.mean())

                    if Y_gene_variance.mean() >= y_var_low and Y_gene_variance.mean() <= y_var_up:
                        print("Y satisfied!")
                        y_satisfied = True
                    elif orthologs:
                        x_satisfied = False  # Dan: starting all over again

                if x_satisfied and y_satisfied:
                    X_final = np.concatenate((og_X, X_core), axis=1)
                    for x in gene_list_x_core:
                        gene_list_x.append(x)
                    with open(file_out1, 'wb') as f:
                        pickle.dump([X_final, data_compounds, gene_list_x, X_gene_variance], f)             
                    print("X genes:", gene_list_x)
                    print("X variance:", X_gene_variance.mean())

                    Y_final = np.concatenate((og_Y, Y_core), axis=1)
                    for x in gene_list_y_core:
                        gene_list_y.append(x)
                    with open(file_out2, 'wb') as f:
                        pickle.dump([Y_final, data_compounds, gene_list_y, Y_gene_variance], f)                    
                    print("Y genes:", gene_list_y)
                    print("Y variance:", Y_gene_variance.mean())


    print("\n\n\nGene selection done.")
            
    #----------------------------------------------
    global x_vivo, y_vivo
    x_vivo = x_type == "rat_vivo"
    y_vivo = y_type == "rat_vivo"
    # ---------------------------------------------

if __name__ == '__main__':
    main()
