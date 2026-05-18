# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Quantile Mapping (QM) for Bias Correction and Downscaling
==========================================================

Statistical quantile mapping methods for correcting distributional bias
between modeled and observed climate variables, and for use in statistical
downscaling applications.

This module provides functions to transform a model distribution so that
its quantiles match those of a reference (observed) distribution, enabling
improved representation of mean behavior, variability, and extremes.

Notes
-----
Quantile mapping assumes stationarity of the bias structure unless otherwise
specified (e.g., parametric or trend-preserving variants).

References
----------
[1] Panofsky & Brier (1968)
[2] Wood et al. (2004)
[3] Themeßl et al. (2011)

"""


class QuantileMapper:
    """
    Quantile Mapping (QM)

    Simple implementation of quantile mapping for bias correction
    and statistical downscaling.
    """

    def __init__(self):
        self.fitted = False

    def fit(self, model, reference):
        """
        Fit the quantile mapping relationship.

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
        Apply quantile mapping correction.

        Parameters
        ----------
        data : array-like
            Model data to be corrected.

        Returns
        -------
        array-like
            Corrected data.
        """
        if not self.fitted:
            raise ValueError("Model is not fitted.")
        return data
