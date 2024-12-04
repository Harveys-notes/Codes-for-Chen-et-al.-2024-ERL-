import netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.rcParams['font.sans-serif'] = "Calibri"
# mpl.rcParams['font.size'] = 6
plt.rc('font',family='Arial',size=8)
# Setting mathfont-------------------------------------
from matplotlib import rcParams
config = {"mathtext.fontset":'stix',}
rcParams.update(config)
# -----------------------------------------------------
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.utils import shuffle
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.dates as mdates
import matplotlib.patches as patches
import os
import math 
from scipy import ndimage
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator 
import matplotlib.dates as mdates
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import cartopy.io.shapereader as shpreader
from pylab import *
import time
import datetime
from scipy import interpolate
import metpy.calc as mpcalc
from metpy.units import units
from skimage import transform
from scipy.stats.mstats import ttest_ind
import cmaps 
plt.close() 




def calc_Ave_weight_lev10to50_neg(time_sta,time_end,lat_sta,lat_end,lon_sta,lon_end,var_name):
    ds_lev10 = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev10.nc')
    T_ano_lev10 = ds_lev10['T_anom'].loc[time_sta:time_end, np.timedelta64(0,'ns'), lat_sta:lat_end, lon_sta:lon_end]
    T_ano_AveContri_lev10 = ds_lev10[var_name].loc[time_sta:time_end, :, lat_sta:lat_end, lon_sta:lon_end].where(T_ano_lev10<0)
    weights = np.cos(np.deg2rad(ds_lev10['lat_traj'].loc[time_sta:time_end, :, lat_sta:lat_end, lon_sta:lon_end])).where(T_ano_lev10<0)
    weights.name = 'weights'
    T_ano_AveContri_lev10_weight = T_ano_AveContri_lev10.weighted(weights.fillna(0)).mean(dim=('time','lat','lon'))

    ds_lev30 = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev30.nc')
    T_ano_lev30 = ds_lev30['T_anom'].loc[time_sta:time_end, np.timedelta64(0,'ns'), lat_sta:lat_end, lon_sta:lon_end]
    T_ano_AveContri_lev30 = ds_lev30[var_name].loc[time_sta:time_end, :, lat_sta:lat_end, lon_sta:lon_end].where(T_ano_lev30<0)
    weights = np.cos(np.deg2rad(ds_lev30['lat_traj'].loc[time_sta:time_end, :, lat_sta:lat_end, lon_sta:lon_end])).where(T_ano_lev30<0)
    weights.name = 'weights'
    T_ano_AveContri_lev30_weight = T_ano_AveContri_lev30.weighted(weights.fillna(0)).mean(dim=('time','lat','lon'))
    
    ds_lev50 = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev50.nc')
    T_ano_lev50 = ds_lev50['T_anom'].loc[time_sta:time_end, np.timedelta64(0,'ns'), lat_sta:lat_end, lon_sta:lon_end]
    T_ano_AveContri_lev50 = ds_lev50[var_name].loc[time_sta:time_end, :, lat_sta:lat_end, lon_sta:lon_end].where(T_ano_lev50<0)
    weights = np.cos(np.deg2rad(ds_lev50['lat_traj'].loc[time_sta:time_end, :, lat_sta:lat_end, lon_sta:lon_end])).where(T_ano_lev50<0)
    weights.name = 'weights'
    T_ano_AveContri_lev50_weight = T_ano_AveContri_lev50.weighted(weights.fillna(0)).mean(dim=('time','lat','lon'))

    T_ano_AveContri_lev10to50_weight = (T_ano_AveContri_lev10_weight + T_ano_AveContri_lev30_weight + T_ano_AveContri_lev50_weight) / 3

    return T_ano_AveContri_lev10to50_weight


def calc_contri_neg(time_sta,time_end,lat_sta,lat_end,lon_sta,lon_end):

    T_ano_AveContri_lev10to50_weight = calc_Ave_weight_lev10to50_neg(time_sta,time_end,lat_sta,lat_end,lon_sta,lon_end,'T_anom')
    seas_AveContri_lev10to50_weight = calc_Ave_weight_lev10to50_neg(time_sta,time_end,lat_sta,lat_end,lon_sta,lon_end,'seas')
    adv_AveContri_lev10to50_weight = calc_Ave_weight_lev10to50_neg(time_sta,time_end,lat_sta,lat_end,lon_sta,lon_end,'adv')
    adiab1_AveContri_lev10to50_weight = calc_Ave_weight_lev10to50_neg(time_sta,time_end,lat_sta,lat_end,lon_sta,lon_end,'adiab1')
    adiab2_AveContri_lev10to50_weight = calc_Ave_weight_lev10to50_neg(time_sta,time_end,lat_sta,lat_end,lon_sta,lon_end,'adiab2')
    adiab3_AveContri_lev10to50_weight = calc_Ave_weight_lev10to50_neg(time_sta,time_end,lat_sta,lat_end,lon_sta,lon_end,'adiab3')
    adiab_AveContri_lev10to50_weight = adiab1_AveContri_lev10to50_weight + adiab2_AveContri_lev10to50_weight + adiab3_AveContri_lev10to50_weight
    diab_AveContri_lev10to50_weight = calc_Ave_weight_lev10to50_neg(time_sta,time_end,lat_sta,lat_end,lon_sta,lon_end,'diab')

    return T_ano_AveContri_lev10to50_weight, seas_AveContri_lev10to50_weight, adv_AveContri_lev10to50_weight, adiab_AveContri_lev10to50_weight, diab_AveContri_lev10to50_weight


def calc_per_live_traj_neg(time_sta,time_end,lat_sta,lat_end,lon_sta,lon_end):
    ds_lev10 = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev10.nc')
    T_ano_lev10 = ds_lev10['T_anom'].loc[time_sta:time_end, np.timedelta64(0,'ns'), lat_sta:lat_end, lon_sta:lon_end]
    ds_lev30 = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev30.nc')
    T_ano_lev30 = ds_lev30['T_anom'].loc[time_sta:time_end, np.timedelta64(0,'ns'), lat_sta:lat_end, lon_sta:lon_end]
    ds_lev50 = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev50.nc')
    T_ano_lev50 = ds_lev50['T_anom'].loc[time_sta:time_end, np.timedelta64(0,'ns'), lat_sta:lat_end, lon_sta:lon_end]

    age_lev10_all = ds_lev10['age'].loc[time_sta:time_end, lat_sta:lat_end, lon_sta:lon_end].where(T_ano_lev10<0).values.flatten()
    age_lev30_all = ds_lev30['age'].loc[time_sta:time_end, lat_sta:lat_end, lon_sta:lon_end].where(T_ano_lev30<0).values.flatten()
    age_lev50_all = ds_lev50['age'].loc[time_sta:time_end, lat_sta:lat_end, lon_sta:lon_end].where(T_ano_lev50<0).values.flatten()

    age_lev10to50_all = np.concatenate([age_lev10_all,age_lev30_all,age_lev50_all])
    age_lev10to50_all = age_lev10to50_all[~np.isnan(age_lev10to50_all)]

    num_all_traj = len(age_lev10to50_all)
    per_live_traj = xr.zeros_like(ds_lev10['T_anom'].mean(dim=('time','lat','lon')), dtype=float)

    for each_value in linspace(0,360,121):
        per_live_traj_ii = np.sum(np.where(age_lev10to50_all<each_value,0,1)) / num_all_traj
        timedelta_neg = -1 * int(each_value)
        per_live_traj.loc[np.timedelta64(timedelta_neg,'h').astype('timedelta64[ns]')] = per_live_traj_ii

    return per_live_traj


time_sta_north_stage1 = '2023-12-10'
time_end_north_stage1 = '2023-12-12'

time_sta_north_stage2 = '2023-12-15'
time_end_north_stage2 = '2023-12-16'

lat_sta_north = 33
lat_end_north = 39
lon_sta_north = 106
lon_end_north = 114

T_ano_AveContri_north_stage1_neg, seas_AveContri_north_stage1_neg, adv_AveContri_north_stage1_neg, adiab_AveContri_north_stage1_neg, diab_AveContri_north_stage1_neg = calc_contri_neg(time_sta_north_stage1, time_end_north_stage1, lat_sta_north, lat_end_north, lon_sta_north, lon_end_north)
per_live_traj_north_stage1_neg = calc_per_live_traj_neg(time_sta_north_stage1, time_end_north_stage1, lat_sta_north, lat_end_north, lon_sta_north, lon_end_north)

T_ano_AveContri_north_stage2_neg, seas_AveContri_north_stage2_neg, adv_AveContri_north_stage2_neg, adiab_AveContri_north_stage2_neg, diab_AveContri_north_stage2_neg = calc_contri_neg(time_sta_north_stage2, time_end_north_stage2, lat_sta_north, lat_end_north, lon_sta_north, lon_end_north)
per_live_traj_north_stage2_neg = calc_per_live_traj_neg(time_sta_north_stage2, time_end_north_stage2, lat_sta_north, lat_end_north, lon_sta_north, lon_end_north)





def calc_traj(time_sta_warming1,time_end_warming1,
              lat_sta_north,lat_end_north,
              lon_sta_north,lon_end_north,
              WorC):
    if WorC == 'warming':
        '''
        Read lat,lon,p of traj
        '''
        ds = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev10.nc')
        T_ano_lev10 = ds['T_anom'].loc[time_sta_warming1:time_end_warming1, np.timedelta64(0,'ns'), lat_sta_north:lat_end_north, lon_sta_north:lon_end_north]
        lev10_lat_traj_north_warming1 = ds['lat_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev10>0)
        lev10_lat_traj_north_warming1_flatten = lev10_lat_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev10_lon_traj_north_warming1 = ds['lon_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev10>0)
        lev10_lon_traj_north_warming1_flatten = lev10_lon_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev10_p_traj_north_warming1 = ds['p_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev10>0)
        lev10_p_traj_north_warming1_flatten = lev10_p_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))

        ds = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev30.nc')
        T_ano_lev30 = ds['T_anom'].loc[time_sta_warming1:time_end_warming1, np.timedelta64(0,'ns'), lat_sta_north:lat_end_north, lon_sta_north:lon_end_north]
        lev30_lat_traj_north_warming1 = ds['lat_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev30>0)
        lev30_lat_traj_north_warming1_flatten = lev30_lat_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev30_lon_traj_north_warming1 = ds['lon_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev30>0)
        lev30_lon_traj_north_warming1_flatten = lev30_lon_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev30_p_traj_north_warming1 = ds['p_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev30>0)
        lev30_p_traj_north_warming1_flatten = lev30_p_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))

        ds = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev50.nc')
        T_ano_lev50 = ds['T_anom'].loc[time_sta_warming1:time_end_warming1, np.timedelta64(0,'ns'), lat_sta_north:lat_end_north, lon_sta_north:lon_end_north]
        lev50_lat_traj_north_warming1 = ds['lat_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev50>0)
        lev50_lat_traj_north_warming1_flatten = lev50_lat_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev50_lon_traj_north_warming1 = ds['lon_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev50>0)
        lev50_lon_traj_north_warming1_flatten = lev50_lon_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev50_p_traj_north_warming1 = ds['p_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev50>0)
        lev50_p_traj_north_warming1_flatten = lev50_p_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))

    elif WorC == 'cooling':
        '''
        Read lat,lon,p of traj
        '''
        ds = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev10.nc')
        T_ano_lev10 = ds['T_anom'].loc[time_sta_warming1:time_end_warming1, np.timedelta64(0,'ns'), lat_sta_north:lat_end_north, lon_sta_north:lon_end_north]
        lev10_lat_traj_north_warming1 = ds['lat_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev10<0)
        lev10_lat_traj_north_warming1_flatten = lev10_lat_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev10_lon_traj_north_warming1 = ds['lon_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev10<0)
        lev10_lon_traj_north_warming1_flatten = lev10_lon_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev10_p_traj_north_warming1 = ds['p_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev10<0)
        lev10_p_traj_north_warming1_flatten = lev10_p_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))

        ds = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev30.nc')
        T_ano_lev30 = ds['T_anom'].loc[time_sta_warming1:time_end_warming1, np.timedelta64(0,'ns'), lat_sta_north:lat_end_north, lon_sta_north:lon_end_north]
        lev30_lat_traj_north_warming1 = ds['lat_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev30<0)
        lev30_lat_traj_north_warming1_flatten = lev30_lat_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev30_lon_traj_north_warming1 = ds['lon_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev30<0)
        lev30_lon_traj_north_warming1_flatten = lev30_lon_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev30_p_traj_north_warming1 = ds['p_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev30<0)
        lev30_p_traj_north_warming1_flatten = lev30_p_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))

        ds = xr.open_dataset('..data/Lagranto-Res/Lagranto-21dayRun/calc_budget/budget_lev50.nc')
        T_ano_lev50 = ds['T_anom'].loc[time_sta_warming1:time_end_warming1, np.timedelta64(0,'ns'), lat_sta_north:lat_end_north, lon_sta_north:lon_end_north]
        lev50_lat_traj_north_warming1 = ds['lat_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev50<0)
        lev50_lat_traj_north_warming1_flatten = lev50_lat_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev50_lon_traj_north_warming1 = ds['lon_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev50<0)
        lev50_lon_traj_north_warming1_flatten = lev50_lon_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))
        lev50_p_traj_north_warming1 = ds['p_traj'].loc[time_sta_warming1:time_end_warming1, :, lat_sta_north:lat_end_north, lon_sta_north:lon_end_north].where(T_ano_lev50<0)
        lev50_p_traj_north_warming1_flatten = lev50_p_traj_north_warming1.stack(all_traj=('time', 'lat', 'lon'))

    return lev10_lon_traj_north_warming1_flatten,lev10_lat_traj_north_warming1_flatten,lev10_p_traj_north_warming1_flatten, \
           lev30_lon_traj_north_warming1_flatten,lev30_lat_traj_north_warming1_flatten,lev30_p_traj_north_warming1_flatten, \
           lev50_lon_traj_north_warming1_flatten,lev50_lat_traj_north_warming1_flatten,lev50_p_traj_north_warming1_flatten


def draw_traj(ii, each_traj, ax, norm,
              lev10_lon_loc_ls, lev10_lat_loc_ls, lev10_p_loc_ls,
              data_ccrs=ccrs.PlateCarree()):
    lev10_lon_nonan = lev10_lon_loc_ls[ii][each_traj,:].dropna(dim='trajtime')
    try:
        if np.any(np.gradient(lev10_lon_nonan.values) >= 90):           # >= because the trajtory in array is reversered (t=-360 to t=0)
            lev10_lon_nonan = xr.where(lev10_lon_nonan<0,lev10_lon_nonan+360,lev10_lon_nonan)
    except:
        error = 0  # nothing happened!!!
    lev10_lat_nonan = lev10_lat_loc_ls[ii][each_traj,:].dropna(dim='trajtime')
    lev10_p_nonan = lev10_p_loc_ls[ii][each_traj,:].dropna(dim='trajtime')
    # 创建每段线的颜色
    colors = [cmaps.MPL_jet(norm(each_p/100)) for each_p in lev10_p_nonan]
    # 创建每段线的线段列表
    segments = [np.column_stack([lev10_lon_nonan[i:i+2].values, lev10_lat_nonan[i:i+2].values]) for i in range(len(lev10_lon_nonan)-1)]
    # 创建每段线的 LineCollection 对象，并指定每段线的颜色
    lcs = [LineCollection([segments[i]], colors=[colors[i]], linewidths=0.4, alpha=0.4, transform=data_ccrs) for i in range(len(segments))]
    for lc in lcs:
        ax.add_collection(lc)


time_sta_north_stage1 = '2023-12-10'
time_end_north_stage1 = '2023-12-12'

time_sta_north_stage2 = '2023-12-15'
time_end_north_stage2 = '2023-12-16' 

lat_sta_north = 33
lat_end_north = 39
lon_sta_north = 106
lon_end_north = 114


[lev10_lon_traj_north_stage1_flatten,lev10_lat_traj_north_stage1_flatten,lev10_p_traj_north_stage1_flatten,
lev30_lon_traj_north_stage1_flatten,lev30_lat_traj_north_stage1_flatten,lev30_p_traj_north_stage1_flatten,
lev50_lon_traj_north_stage1_flatten,lev50_lat_traj_north_stage1_flatten,lev50_p_traj_north_stage1_flatten] = calc_traj(time_sta_north_stage1,time_end_north_stage1,
                                                                                                                    lat_sta_north,lat_end_north,
                                                                                                                    lon_sta_north,lon_end_north,
                                                                                                                    WorC='cooling')
[lev10_lon_traj_north_stage2_flatten,lev10_lat_traj_north_stage2_flatten,lev10_p_traj_north_stage2_flatten,
lev30_lon_traj_north_stage2_flatten,lev30_lat_traj_north_stage2_flatten,lev30_p_traj_north_stage2_flatten,
lev50_lon_traj_north_stage2_flatten,lev50_lat_traj_north_stage2_flatten,lev50_p_traj_north_stage2_flatten] = calc_traj(time_sta_north_stage2,time_end_north_stage2,
                                                                                                                    lat_sta_north,lat_end_north,
                                                                                                                    lon_sta_north,lon_end_north,
                                                                                                                    WorC='cooling')


lev10_lon_flatten_ls = [lev10_lon_traj_north_stage1_flatten.transpose(), lev10_lon_traj_north_stage2_flatten.transpose()]
lev10_lat_flatten_ls = [lev10_lat_traj_north_stage1_flatten.transpose(), lev10_lat_traj_north_stage2_flatten.transpose()]
lev10_p_flatten_ls = [lev10_p_traj_north_stage1_flatten.transpose(), lev10_p_traj_north_stage2_flatten.transpose()]

lev30_lon_flatten_ls = [lev30_lon_traj_north_stage1_flatten.transpose(), lev30_lon_traj_north_stage2_flatten.transpose()]
lev30_lat_flatten_ls = [lev30_lat_traj_north_stage1_flatten.transpose(), lev30_lat_traj_north_stage2_flatten.transpose()]
lev30_p_flatten_ls = [lev30_p_traj_north_stage1_flatten.transpose(), lev30_p_traj_north_stage2_flatten.transpose()]

lev50_lon_flatten_ls = [lev50_lon_traj_north_stage1_flatten.transpose(), lev50_lon_traj_north_stage2_flatten.transpose()]
lev50_lat_flatten_ls = [lev50_lat_traj_north_stage1_flatten.transpose(), lev50_lat_traj_north_stage2_flatten.transpose()]
lev50_p_flatten_ls = [lev50_p_traj_north_stage1_flatten.transpose(), lev50_p_traj_north_stage2_flatten.transpose()]



import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection


fig, axs = plt.subplots(2, 3, figsize=(6*1.2,4*1.2), gridspec_kw={'width_ratios':[2,2.75,1.25]}, dpi=1000)

title_ls = ['(a)', '(b)']
projection = ccrs.Orthographic(75,45)
data_ccrs = ccrs.PlateCarree()
for ii,ax in enumerate(axs[:,0]):
    ax.axis('off')
    ax = fig.add_subplot(2, 3, (ii+1)*3-2,projection=projection)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.3, edgecolor='black') 
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#dde5f2') 

    gl = ax.gridlines(draw_labels=False, x_inline=False, y_inline=False, linestyle='--', color='grey', linewidth=0.15, alpha=1)
    gl.xlocator = mticker.FixedLocator(np.arange(-180,420,60))
    gl.ylocator = mticker.FixedLocator(np.arange(0,120,30))

    ax.set_title(title_ls[ii], loc='left')

    ### Traj ###
    norm = mcolors.Normalize(vmin=500, vmax=1000)
    for each_traj in range(lev10_lon_flatten_ls[ii].shape[0]):
        ### ------------------------------------------lev10------------------------------------------ ###
        draw_traj(ii, each_traj, ax, norm, lev10_lon_flatten_ls, lev10_lat_flatten_ls, lev10_p_flatten_ls)
        ### ------------------------------------------lev30------------------------------------------ ###
        draw_traj(ii, each_traj, ax, norm, lev30_lon_flatten_ls, lev30_lat_flatten_ls, lev30_p_flatten_ls)
        # ### ------------------------------------------lev50------------------------------------------ ###
        draw_traj(ii, each_traj, ax, norm, lev50_lon_flatten_ls, lev50_lat_flatten_ls, lev50_p_flatten_ls)
        print(each_traj)

    ### Rectangle ###
    rectangle_north = patches.Rectangle((106, 33), 8, 6, edgecolor='purple', facecolor='none', lw=0.8, alpha=1, transform=data_ccrs, zorder=7)
    rectangle_south = patches.Rectangle((104, 25), 12, 6, edgecolor='purple', facecolor='none', lw=0.8, alpha=1, transform=data_ccrs, zorder=7)
    ax.add_patch(rectangle_north)
    # ax.add_patch(rectangle_south)

    ax.set_global()     # Ensure to show the whole earth

### Set the colorbar ###
norm = mcolors.Normalize(vmin=500, vmax=1000)
ax_cb = ax.inset_axes([0, -0.4, 1, 0.075])    #creat a subplot for colobar
                                                    #[left starting point, bottom starting point, length, width] unit:pixels
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmaps.MPL_jet),cax=ax_cb,orientation='horizontal',spacing='proportional')
cb.ax.tick_params(axis='both', length=0)
cb.outline.set_visible(True)
cb.set_label('unit: hPa')
cb.outline.set_linewidth(0.3)
cb.set_ticks(np.linspace(500,1000,5))


title_ls = ['(c)', '(d)']
T_ano_AveContri_ls = [T_ano_AveContri_north_stage1_neg, T_ano_AveContri_north_stage2_neg]
seas_AveContri_ls = [seas_AveContri_north_stage1_neg, seas_AveContri_north_stage2_neg]
adv_AveContri_ls = [adv_AveContri_north_stage1_neg, adv_AveContri_north_stage2_neg]
adiab_AveContri_ls = [adiab_AveContri_north_stage1_neg, adiab_AveContri_north_stage2_neg]
diab_AveContri_ls = [diab_AveContri_north_stage1_neg, diab_AveContri_north_stage2_neg]
per_live_traj_ls = [per_live_traj_north_stage1_neg, per_live_traj_north_stage2_neg]
for ii,ax in enumerate(axs[:,1]):
    ### Twin axis-1 --------------------------------------------------------------- ###
    # line1, = ax.plot(np.linspace(-360,0,121), T_ano_AveContri_ls[ii], label=r'$\mathbf{T^{\prime}}$', 
    #                      alpha=0.8, lw=3.5, color='black', zorder=3)
    line1, = ax.plot(np.linspace(-360,0,121), seas_AveContri_ls[ii]+adv_AveContri_ls[ii]+adiab_AveContri_ls[ii]+diab_AveContri_ls[ii], 
                     label=r'$\mathbf{T^{\prime}}$', alpha=0.8, lw=3.5, color='black', zorder=3)
    line2, = ax.plot(np.linspace(-360,0,121), seas_AveContri_ls[ii], label=r'$\mathbf{Seasonality\; T^{\prime}}$', color='tab:blue', alpha=0.8, lw=1.8, zorder=5)
    line3, = ax.plot(np.linspace(-360,0,121), adv_AveContri_ls[ii], label=r'$\mathbf{Advective\; T^{\prime}}$', color='tab:orange', alpha=0.8, lw=1.8, zorder=5)
    line4, = ax.plot(np.linspace(-360,0,121), adiab_AveContri_ls[ii], label=r'$\mathbf{Adiabatic\; T^{\prime}}$', color='tab:green', alpha=0.8, lw=1.8, zorder=5)
    line5, = ax.plot(np.linspace(-360,0,121), diab_AveContri_ls[ii], label=r'$\mathbf{Diabatic\; T^{\prime}}$', color='tab:red', alpha=0.8, lw=1.8, zorder=5)

    ### y-axis ###
    ax.set_yticks(np.linspace(-18,18,5))
    ax.set_ylim(-18,18)
    ax.set_ylabel('Temperature anomaly (K)')

    ### x-axis ###
    ax.set_xticks(np.linspace(-360,0,6))
    ax.set_xlim(-360-36,0+36)
    ax.set_xlabel('Trajectory time (hour)')

    ax.grid(True, lw=0.3, ls='--')
    ax.set_title(title_ls[ii], loc='left')
    ### --------------------------------------------------------------------------- ###


    ### Twin axis-2 --------------------------------------------------------------- ###
    ax2 = ax.twinx()
    ax2.fill_between(np.linspace(-360,0,121),per_live_traj_ls[ii]*100, 
                    facecolor='gray', alpha=0.3, zorder=0)
    ax2.set_yticks(np.linspace(0,100,5))
    ax2.set_ylim(-100,100)
    ### --------------------------------------------------------------------------- ###
legend = fig.legend(handles=[line1,line2,line3,line4,line5], loc='lower center', fontsize=8, ncol=5,  bbox_to_anchor=(0.675, 0.07))
legend.get_frame().set_alpha(0)


title_ls = ['(e)', '(f)']
for ii,ax in enumerate(axs[:,2]):
    ### ax --------------------------------------------------------------- ###
    width = 0.75
    bar_color = ['black','tab:blue','tab:orange','tab:green','tab:red'][::-1]
    # terms_contri = np.asarray([T_ano_AveContri_ls[ii][-1].values,seas_AveContri_ls[ii][-1].values,adv_AveContri_ls[ii][-1].values,adiab_AveContri_ls[ii][-1].values,diab_AveContri_ls[ii][-1].values][::-1])
    terms_contri = np.asarray([seas_AveContri_ls[ii][-1].values+adv_AveContri_ls[ii][-1].values+adiab_AveContri_ls[ii][-1].values+diab_AveContri_ls[ii][-1].values,
                               seas_AveContri_ls[ii][-1].values,adv_AveContri_ls[ii][-1].values,adiab_AveContri_ls[ii][-1].values,diab_AveContri_ls[ii][-1].values][::-1])
    terms_name = [r'$\mathbf{T^{\prime}}$', r'$\mathbf{Seasonality\; T^{\prime}}$', r'$\mathbf{Advective\; T^{\prime}}$', r'$\mathbf{Adiabatic\; T^{\prime}}$', r'$\mathbf{Diabatic\; T^{\prime}}$'][::-1]
    x_ticks = np.arange(len(terms_contri))
    
    rects = ax.barh(x_ticks, terms_contri, height=width, color=bar_color)
    ax.bar_label(rects, padding=1, fmt='%2.1f', fontsize=6, fontweight='bold')
    
    ax.set_xticks(np.linspace(-18,18,5))
    ax.set_xlim(-20,20)
    ax.set_xlabel('Temperature anomaly (K)')
    ax.axvline(x=0, lw=0.8, linestyle='-', color='black')

    ax.set_yticks(x_ticks, terms_name)
    # 隐藏右边框线和上边框线
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title(title_ls[ii], loc='left')
    ### --------------------------------------------------------------------------- ###


plt.tight_layout()
plt.savefig(f'figure-3.jpg', bbox_inches='tight')
plt.show()