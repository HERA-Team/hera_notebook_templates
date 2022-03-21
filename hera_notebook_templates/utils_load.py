# -*- coding: utf-8 -*-
# Copyright 2022 the HERA Project
# Licensed under the MIT License
# Written by Dara Storer - dstorer@uw.edu

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pyuvdata import UVCal, UVData, UVFlag, utils
import os
import sys
import glob
import uvtools as uvt
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, Angle
from astropy.coordinates import SkyCoord as sc
import pandas
import warnings 
import copy
from hera_mc import cm_hookup, geo_sysdef
import math
from uvtools import dspec
import hera_qm 
from hera_mc import cm_active
from matplotlib.lines import Line2D
from matplotlib import colors, cm
import json
from hera_notebook_templates.data import DATA_PATH
from astropy.io import fits
import csv
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic
import healpy
from multiprocessing import Process, Queue
from bokeh.layouts import row, column
from bokeh.models import CustomJS, Select, RadioButtonGroup, Range1d
from bokeh.plotting import figure, output_file, show, ColumnDataSource
import scipy
from hera_notebook_templates import utils
warnings.filterwarnings('ignore')

# useful global variables
status_colors = {
    'dish_maintenance' : 'salmon',
    'dish_ok' : 'red',
    'RF_maintenance' : 'lightskyblue',
    'RF_ok' : 'royalblue',
    'digital_maintenance' : 'plum',
    'digital_ok' : 'mediumpurple',
    'calibration_maintenance' : 'lightgreen',
    'calibration_ok' : 'green',
    'calibration_triage' : 'lime',
    'not_connected' : 'black'}
status_abbreviations = {
    'dish_maintenance' : 'dish-M',
    'dish_ok' : 'dish-OK',
    'RF_maintenance' : 'RF-M',
    'RF_ok' : 'RF-OK',
    'digital_maintenance' : 'dig-M',
    'digital_ok' : 'dig-OK',
    'calibration_maintenance' : 'cal-M',
    'calibration_ok' : 'cal-OK',
    'calibration_triage' : 'cal-Tri',
    'not_connected' : 'NC'}

