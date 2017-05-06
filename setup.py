#!/usr/bin/env python
from __future__ import print_function
"""
Processing: Software for time-series analysis, spectral analysis and
transfer function computation from geophysical data.
"""

from distutils.core import setup
from setuptools import find_packages

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Natural Language :: English',
]

with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

setup(
    name="processing",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.7',
        'scipy>=0.13',
        'matplotlib',
        'properties[math]'
    ],
    author="SimPEG Team",
    author_email="lindseyheagy@gmail.com",
    description="Processing: time series, spectral analysis and computation of transfer functions from geophysical data",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="geophysics, time series, spectral analysis",
    url="http://simpeg.xyz",
    download_url="http://github.com/simpeg/processing",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False
)
