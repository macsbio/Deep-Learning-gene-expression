#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
from typing import List, Dict

import numpy as np

ColInfo = collections.namedtuple("ColInfo", ['compound', 'dosage', 'time', 'replicate'])
ColInfo.__doc__ = """\
Contains indices of compound, dosage, time and replicate of the corresponding data column.
"""

MinMax = collections.namedtuple("MinMax", ['min', 'max'])
MinMax.__doc__ = """\
Contains two floats, one representing the minimum and one the maximum of a dataset.
"""


class Header:
    """
    Contains different header information.

    Attributes:
        compounds: List[str] of compound names
        dosages: List[str] of dosage names
        times: List[str] of time names
        replicates: List[str] of replicates (1, 2 etc. but as strings)
        columns: List[ColInfo] containing indices for the above lists
        genes: List[str] of the names of the genes for each row
    """

    def __init__(self, compounds: List[str], dosages: List[str], times: List[str], replicates: List[str],
                 columns: List[ColInfo], genes: List[str] = None):
        self.compounds = compounds
        self.dosages = dosages
        self.times = times
        self.replicates = replicates
        # {'compound': comp_index, 'dosage': dos_index, 'time': time_index, 'replicate': repl_index}
        self.columns = columns
        self.genes, self._gene_indices = None, None
        self.set_genes(genes)

    def set_genes(self, genes: List[str]) -> None:
        """Set genes and generate indices"""
        self.genes = genes if genes else []
        self._gene_indices = self.__generate_gene_indices(self.genes)

    def get_gene_indices(self, name: str) -> List[int]:
        """Returns list of row indices for gene name"""
        return self._gene_indices[name]

    def copy(self):
        """Return a shallow copy of this header."""
        return Header(self.compounds, self.dosages, self.times, self.replicates, self.columns, self.genes)

    @staticmethod
    def __generate_gene_indices(genes: List[str]) -> Dict[str, List[int]]:
        gene_indices = {}
        for i, name in enumerate(genes):
            if name not in gene_indices:
                gene_indices[name] = []
            gene_indices[name].append(i)
        return gene_indices

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.compounds == other.compounds and \
                   self.dosages == other.dosages and \
                   self.times == other.times and \
                   self.replicates == other.replicates and \
                   self.columns == other.columns and \
                   self.genes == other.genes
        else:
            return False

    def __str__(self):
        return "Compounds: {}\nDosages: {}\nTimepoints: {}\nReplicates: {}\nGenes: {}".format(self.compounds,
                                                                                              self.dosages,
                                                                                              self.times,
                                                                                              self.replicates,
                                                                                              self.genes)


class Data:
    """Represents Data read from a single file and convenience methods to access those."""

    def __init__(self, header: Header, activations: np.ndarray):
        self.header = header
        self.activations = activations
        self.hierarchical = None
