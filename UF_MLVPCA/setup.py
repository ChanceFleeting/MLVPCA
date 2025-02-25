#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on %(date)s

@author: Chance Fleeting

This code is formatted in accordance with PEP8
See: https://peps.python.org/pep-0008/

use %matplotlib qt to display images in pop out
use %matplotlib inline to display images inline

!python setup.py build_ext --inplace --verbose

"""
__author__ = 'Chance Fleeting'
__version__ = '0.0'

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext  # Corrected import
import pybind11
import traceback  # For error logging
import os, sys

# Specify the local Armadillo path
armadillo_path = './armadillo'  # Replace with the relative path to your local Armadillo directory

# Error log file
error_log_file = 'setup_error.log'

try:
    # Define paths for Armadillo
    armadillo_path = './armadillo'  # Adjust to the correct location if needed
    include_dir = os.path.join(armadillo_path, 'include')
    lib_dir = os.path.join(armadillo_path, 'lib')
    
    # Validate Armadillo path
    if not os.path.isdir(armadillo_path):
        raise FileNotFoundError(f"Armadillo path not found: {armadillo_path}")
    if not os.path.isdir(include_dir):
        raise FileNotFoundError(f"Include directory not found: {include_dir}")
    if not os.path.isdir(lib_dir):
        raise FileNotFoundError(f"Library directory not found: {lib_dir}")
    
    # Define the extension module
    ext_modules = [
        Extension(
            'fun_GetCov_SeqADMM_SelectTuningPar',
            ['fun_GetCov_SeqADMM_SelectTuningPar_c.cpp'],
            include_dirs=[
                pybind11.get_include(),
                include_dir,
                "C:/Users/cflee/anaconda3/Library/include",  # Conda OpenBLAS and LAPACK include
            ],
            library_dirs=[
                lib_dir,
                "C:/Users/cflee/anaconda3/Library/lib",  # Conda OpenBLAS and LAPACK lib
            ],
            libraries=['armadillo', 'openblas', 'lapack'],
            language='c++',
            extra_compile_args=['-O3'],  # Removed `-std=c++11` for compatibility, add if needed
        ),
    ]
    
    # Run setup
    setup(
        name='fun_GetCov_SeqADMM_SelectTuningPar',
        version='0.2',
        author='Chance Fleeting',
        ext_modules=ext_modules,
        cmdclass={'build_ext': build_ext},  # Ensure pybind11 integration
    )

except Exception as e:
    # Log the error to the file
    with open(error_log_file, 'w') as log_file:
        log_file.write("An error occurred during setup:\n")
        log_file.write(traceback.format_exc())
    print(f"An error occurred during setup. Details have been logged to {error_log_file}")
