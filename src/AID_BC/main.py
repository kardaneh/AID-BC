#!/usr/bin/env python3
# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
CLI entry point
===============

Command-line interface for bias correction and downscaling methods
(e.g., Quantile Mapping, Optimal Transport) applied to climate data.
"""

import argparse


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Bias correction and downscaling CLI")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["qm", "ot"],
        help="Method to use: qm (Quantile Mapping) or ot (Optimal Transport)",
    )

    return parser.parse_args()


def main():
    """
    Main CLI entry point.
    """
    args = parse_args()

    if args.method == "qm":
        print("Running Quantile Mapping...")
    elif args.method == "ot":
        print("Running Optimal Transport...")


if __name__ == "__main__":
    main()
