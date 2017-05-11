# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:00:19 2017

@author: jpeacock
"""
#==============================================================================
# Imports
#==============================================================================
import os
import numpy as np

#==============================================================================
# Variables
#==============================================================================
# path to time series files
ts_path = os.getcwd()

data_type = np.dtype([('station', '|S10'),
                      ('comp', '|S3'),
                      ('sampling_rate', np.float),
                      ('start_time', np.float),
                      ('npts', np.int),
                      ('units', '|S10'),
                      ('lat', np.float),
                      ('lon', np.float),
                      ('elev', np.float),
                      ('fn', '|S100')])

#==============================================================================
# get list of time series files
ts_fn_list = [os.path.join(ts_path, ts_fn) for ts_fn in os.listdir(ts_path)
              if ts_fn[-2:] in ['EX', 'EY', 'HX', 'HY']]

# get information from time series to make sure they are all lined up
# for now get all the header information into a numpy array where the 
# data type is the header information
ts_header_arr = np.zeros(len(ts_fn_list), 
                         dtype=data_type)

# read in the header from each file and put it in the array
for ii, ts_fn in enumerate(ts_fn_list):
    with open(ts_fn, 'r') as fid:
        header = fid.readline()[1:].strip().split()
        
    # append the file name for convenience later
    header.append(ts_fn)
     
    # put the information into the array accordingly
    for h_name, info in zip(data_type.names, header):
        ts_header_arr[ii][h_name] = info

# sort out to make sure all time series have the same station name, 
# start time and sampling rate




        
    