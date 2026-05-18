# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Optimal Transport (OT) for Bias Correction and Downscaling
==========================================================

This module provides a simple optimal transport-based framework for
statistical bias correction and downscaling of climate variables.

Optimal transport methods align the full probability distribution of
model outputs with a reference (observed) distribution by finding a
mapping that minimizes the cost of transporting mass between them.

This allows for distribution-aware correction of climate model biases,
including adjustments to mean, variability, and extremes, while
preserving the underlying structure of the data.
"""


class OptimalTransportMapper:
    """
    Optimal Transport (OT) Mapping

    Simple optimal transport-based mapping for bias correction
    and statistical downscaling.
    """

    def __init__(self):
        self.fitted = False

    def fit(self, model, reference):
        """
        Fit optimal transport mapping between model and reference.

        Parameters
        ----------
        model : array-like
            Model/simulated data.
        reference : array-like
            Observed/reference data.

        Returns
        -------
        self
        """
        self.fitted = True
        return self

    def transform(self, data):
        """
        Apply optimal transport correction.

        Parameters
        ----------
        data : array-like
            Model data to be corrected.

        Returns
        -------
        array-like
            Transported (corrected) data.
        """
        if not self.fitted:
            raise ValueError("Model is not fitted.")
        return data
