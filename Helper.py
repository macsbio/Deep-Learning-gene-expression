#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import numpy as np
import tensorflow as tf
from keras.metrics import mse, mae, mape

def add_once_to_list(item, coll: List) -> int:
    """Add item to list if it is not yet contained, and return its index."""
    if item not in coll:
        coll.append(item)
    return coll.index(item)


def create_gene_name_mapping(gene_list=None, domain="rat_vivo"):  
    """
    Creates a mapping for Gene.Symbol entries for rat in vitro/in vivo data (i.e. Fabp4) to gene names in human in vitro data set (i.e. 
    FABP4).
    """
    if gene_list is None:
        genes_small = ['FABP4', 'ACACA', 'AKT1', 'AKT2', 'AKT3', 'PRKAA1', 'PRKAA2', 'ADIPOR1', 'ADIPOR2', 'ADIPOQ',
                        'BCL2A1', 'CPT2', 'CPT1A', 'CPT1B', 'CASP8', 'MLXIPL', 'FABP5', 'ELOVL3', 'FAS', 'FOXO1', 'NR1H4',
                        'RXRA', 'FASLG', 'FABP3', 'FABP7', 'PMP2', 'GCKR', 'IL1A', 'IL10', 'IRS1', 'IRS2', 'MAPK10', 'NFKB1',
                        'NFKB2', 'RELA', 'RELB', 'PPARA', 'PPARG', 'PTEN', 'RXRB', 'RXRG', 'SCD', 'SOCS3', 'SREBF1', 'TGFB1',
                        'TGFB2', 'TGFB3', 'TLR4', 'TNF', 'PNPLA3', 'MTOR']
    else:
        genes_small = gene_list

    gene_mapping = {g.capitalize(): g for g in genes_small}  #human data are all caps
    if domain=="human_vitro":
        gene_mapping = {g.upper(): g for g in genes_small}  #I added this
    
    # manually accounting for a subset of genes
    try:
        gene_mapping['Scd1'] = gene_mapping['Scd']
        del gene_mapping['Scd']
        del gene_mapping['Tnf']
    except KeyError:
        pass


    # manually accounting for a subset of genes from the Cholestasis gene list
    try:
        gene_mapping['Slco1b2'] = gene_mapping['Slco1b1']
        del gene_mapping['Slco1b1']
        # gene_mapping['Slco1b2'] = gene_mapping['Slco1b3']
        #del gene_mapping['Slco1b3']  # Dan: I commented this out because otherwise the number of genes in both domains are not equal!

        gene_mapping['Ugt2b35'] = gene_mapping['Ugt2b4']
        del gene_mapping['Ugt2b4']

        gene_mapping['Cyp3a18'] = gene_mapping['Cyp3a4']
        del gene_mapping['Cyp3a4']
    except KeyError:
        pass

    # manually accounting for a subset of genes from the Carcinogenic gene list
    try:
        gene_mapping['Fibcd1'] = gene_mapping['Fcn3']
        del gene_mapping['Fcn3']
    except KeyError:
        pass
#    print(gene_mapping.keys())
    return gene_mapping

