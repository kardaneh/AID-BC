# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Models for bias correction"""

from AID_BC.models.qm import QuantileMapper
from AID_BC.models.ot import OptimalTransportMapper

__all__ = [
    "QuantileMapper",
    "OptimalTransportMapper",
]
