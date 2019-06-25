# Deep-Learning-gene-expression

Predicting gene expression patterns across different domains:
Predicting time series of human in vitro and rat in vivo gene expressing given a measured time series of rat in vitro gene expression following exposure to a previously unseen compound. 

Multiple deep learning appraoched (Convolutional Neural Network, A bottleneck Artificial Neural Network with bottleneck architecutre, and a Modifed Autoencoder approach) are compared to traditional machine learning techniques (k-Nearest Neighbours, random regression forest) in predicting gene expression patterns acros domains. 
The deep learning models are implemented using Keras. Tranditional machine learning models have been implemented using sklearn. Scripts to read and parse the time series of gene expression data are also supplied. Time series of rat in vitro, human in vitro, and rat in vivo micro-array gene expression data from Open TG-GATEs, a large publically avaliable toxicogenomics data base [1], are used to generate machine learning examples.

This project is the code refernced to in "Use of deep learning methods to translate drug-induced gene expression changes from rat to human primary hepatocyte exposed in vitro and in vivo" by 
Shauna O’Donovan, Kurt Driessens, Daniel Lopatta, Florian Wimmenauer, Alexander Lukas, Jelmer Neeven, Tobias Stumm, Evgueni Smirnov, Michael Lenz, Gokhan Ertaylan, Danyel Jennen, Natal van Riel, Rachel Cavill, Ralf Peeters, Theo de Kok.

# Dependancies 

The scripts are implemented in Python 3, Keras is required to run the scripts. 

# How to run

Sample data files can be downloaded from the following link.

https://surfdrive.surf.nl/files/index.php/s/dyVphSI1xXS8Zxi

This folder contains three files containing processed rat in vitro, human in vitro, and rat in vivo gene expression data obtained from open TG-GATEs [1]. The original raw data micro-array data can be downloaded in the form of CEL files from https://toxico.nibiohn.go.jp.
The micro-array data has been pre-processed using Affymetrix Power Tools using the robust multi-array average normalisation method and stored in the form of pickle files as follows:

data[gene][compound][dosage][replicate][time]

Where genes are indicated by name (gene symbol). Compound, dosage, replicate, and time can be indicated by index.

Train models using provided toxicologicaly relevent gene sets (i.e. NAFLD, STEATOSIS etc.):
-	Call file PredictEncoding_preselected.py from job script
-	Scroll to bottom of PredictEncoding_preselected (“main”)
-	Change gene list as desired (CHOLESTASIS, NAFLD, STEATOSIS, GTX_CAR)
-	Change domain and method as desired (predicting human in vitro or rat in vivo)
- Specify model to be trained (cnn, naive_encoder, mod_autoencoder, rrf, knn)

Creating new gene sets:
-	Call file RandomGenes.py from job script
-	Change parameters in RandomGenes.py as desired (specifiy nested or independent, orthologs or non-orthologs).

Existing sets (i.e. random, nested or orthologs):
-	Call file PredictEncoding.py from job script
-	Scroll to bottom of PredictEncoding (“main”)
-	Specify the output domain (human vitro, rat vivo)
-	Change parameters of loops as desired
-	Specify gene set to be used - preselected orthologus gene sets
-	Specify prediction method as desired: naive_encoder, mod_autoencoder, cnn, or rf, and knn.

Other remarks:
-	In order to investigate a data file, use OpenData.sh and OpenData.py
-	data files outputed from RandomGenes.py will have format
[og_X, data_compounds, gene_list_x, gene_variance],
where og_X are the actual gene expression values


#References
1.	Igarashi, Y., Nakatsu, N., Yamashita, O, Ono, A., Urushidani, T., Yamada, H., Open TG-GATEs: a large-scale toxicogenomics database. Nucleic Acids Res. 43, 21-7.

