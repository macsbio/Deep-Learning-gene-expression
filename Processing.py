#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import tensorflow as tf
from typing import List, Dict, Callable, Tuple

from Helper import create_gene_name_mapping

import numpy as np

from Types import Data


def get_hierarchical_data(data: Data, genes_to_use: List[str] = None,
                          multigene_reducer: Callable[[List[np.array]], np.array] = multigenes_median
                          ) -> Dict[str, List[List[List[List[float]]]]]:
    """
    Data formatted as data[gene][compound][dosage][replicate][time]
                           name  index     index   index      index
    """

    activations = data.activations
    header = data.header

    if genes_to_use:
        considered_genes = list(set(header.genes).intersection(set(genes_to_use)))
        if len(considered_genes) != len(genes_to_use):
            print("WARN: {} genes specified, but only {} found in intersection".format(len(genes_to_use),
                                                                                       len(considered_genes)))
            print("Namely ", set(genes_to_use)-set(considered_genes), " is/are missing")
    else:
        considered_genes = list(set(header.genes))

    hierarchical = \
        {gene: [[[[]
                  for _ in header.replicates]
                 for _ in header.dosages]
                for _ in header.compounds]
         for gene in considered_genes}
    cols = header.columns

    for gene in [g for g in considered_genes if g is not None]:

        indices_for_gene = header.get_gene_indices(gene)
        gene_activs = [activations[idx] for idx in indices_for_gene]
        activ = multigene_reducer(gene_activs)

        for i in range(len(activ)):
            hierarchical[gene][cols[i].compound][cols[i].dosage][cols[i].replicate].append(activ[i])
    return hierarchical


def correct_slopes(X: np.array, vivo=False) -> np.array:
    """ Change Learning data to shape of series instead of series """
    if not vivo:
        for i in range(0, X.shape[1], 3):
            X[:, i + 2] = X[:, i + 2] - X[:, i + 1]
            X[:, i + 1] = X[:, i + 1] - X[:, i + 0]
    else:
        for i in range(0, X.shape[1], 4):
            X[:, i + 3] = X[:, i + 3] - X[:, i + 2]
            X[:, i + 2] = X[:, i + 2] - X[:, i + 1]
            X[:, i + 1] = X[:, i + 1] - X[:, i + 0]
    return X

def decorrect_slopes(X: np.array, vivo=False) -> np.array:
    """ Change slope of series back to original data """
    if not vivo:
        for i in range(0, X.shape[1], 3):
            X[:, i + 1] = X[:, i + 1] + X[:, i + 0]
            X[:, i + 2] = X[:, i + 2] + X[:, i + 1]
    else:
        for i in range(0, X.shape[1], 4):
            X[:, i + 1] = X[:, i + 1] + X[:, i + 0]
            X[:, i + 2] = X[:, i + 2] + X[:, i + 1]
            X[:, i + 3] = X[:, i + 3] + X[:, i + 2]
    return X

def normalize_total(X: np.array) -> Tuple[np.array, Tuple]:
    """ Normalize using min/max of entire dataset"""
    maxX = np.max(X)
    minX = np.min(X)
    X = (X - minX) / (maxX - minX)
    return X, (minX, maxX)

def standardize_features(X: np.array) -> Tuple[np.array, Tuple]:
    """ Standardize per feature."""
    mean = np.mean(X, axis=0)
    corrected = X - mean
    std = np.std(corrected, axis=0)
    standardized = corrected / std
    return standardized, (mean, std)

def destandardize(X: np.array, stats: Tuple) -> np.array:
    """ Destandardize using (mean, std) per feature."""
    X *= stats[1]
    X += stats[0]
    return X

def standardize_with_stats(X: np.array, stats: Tuple) -> np.array:
    """ Standardize using (mean, std) per feature."""
    corrected = X - stats[0]
    standardized = corrected / stats[1]
    return standardized

def normalize_features(X: np.array) -> Tuple[np.array, Tuple]:
    """ Normalize using min/max per feature."""
    maxX = np.max(X, axis=0)
    minX = np.min(X, axis=0)
    X = (X - minX) / (maxX - minX)
    return X, (minX, maxX)

def normalize_with_norms(X: np.array, norms: Tuple) -> np.array:
    """ Normalize using given norms. Tuple is (min, max) where min 
    and max are either integers or np.arrays (when normalizing per feature)"""
    X = (X - norms[0]) / (norms[1] - norms[0])
    return X


def denormalize_with_norms(X: np.array, norms: Tuple) -> np.array:
    """ Denormalize using given norms. Tuple is (min, max) where min 
    and max are either integers or np.arrays (when normalizing per feature)"""
    X = (X * (norms[1] - norms[0])) + norms[0]
    return X
