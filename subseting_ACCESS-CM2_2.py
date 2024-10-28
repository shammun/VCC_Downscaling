import os
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from google.colab import drive
import geopandas as gpd

# Mount Google Drive
drive.mount('/content/drive')

# Set the base path for your project in Google Drive
base_path = '/content/drive/My Drive/VCC'  # Replace with your actual folder path

# Read input files
future_dates = pd.read_csv(f'{base_path}/future_dates_2023.csv')
historical_dates = pd.read_csv(f'{base_path}/historical_dates_2023.csv')
models = pd.read_csv(f'{base_path}/models_2023.csv')
variables = pd.read_csv(f'{base_path}/Variables.csv')
ssps = ["historical", "ssp245", "ssp370", "ssp585"]

# Adjust historical_days to start from 1981 and end at 2014
historical_days = np.arange(1, 12411)  # 34 years * 365 days = 12410 days
# Adjust ssp_days to start from 2015 and end at 2100
ssp_days = np.arange(12411, 43801)  # 86 years * 365 days = 31390 days, starting from 12411

# Model specific information
m = 11  # Model index
model = models.iloc[m-1, 0]
realization = models.iloc[m-1, 3]
grid = models.iloc[m-1, 4]
guide = gpd.read_file(f'{base_path}/grids/subsets/{model}_guide.shp')
lon_res = models.iloc[m-1, 24]  # pixel resolutions
lat_res = models.iloc[m-1, 23]

def process_historical(var, ssp, dates):
    total_days = 12410
    pixels = np.full((total_days, len(guide)), -999999, dtype=np.float32)
    
    start_index = 0
    for d in range(7):  # 7 chunks of 5 years each
        date = dates[d]
        print(f"Processing {date}")
        nc_name = f'{base_path}/cmip6/{var}/{model}/{ssp}/ClimAVA-SW_{model}_{ssp}_{var}_{realization}_{date}.nc'
        # nc_name = f'{base_path}/cmip6/{var}/{model}/{ssp}/{var}_day_{model}_{ssp}_{realization}_{date}.nc'
        
        with xr.open_dataset(nc_name) as nc:
            array = nc[var].values
            chunk_days = array.shape[2]  # Get the number of days in this chunk

            for p, (lat, lon) in enumerate(zip(guide['lat'], guide['lon'])):
                Y = int(((lat + 90) / lat_res) + 1)
                X = int((lon / lon_res) + 1)
                pixels[start_index:start_index+chunk_days, p] = array[X, Y, :]

        start_index += chunk_days

    return pixels

def process_future(var, ssp, dates):
    total_days = 31390
    pixels = np.full((total_days, len(guide)), -999999, dtype=np.float32)
    
    start_index = 0
    for d in range(17):  # 18 chunks of 5 years each (17 full chunks + 1 partial for 2100)
        date = dates[d]
        print(f"Processing {date}")
        nc_name = f'{base_path}/cmip6/{var}/{model}/{ssp}/ClimAVA-SW_{model}_{ssp}_{var}_{realization}_{date}.nc'
        # nc_name = f'{base_path}/cmip6/{var}/{model}/{ssp}/{var}_day_{model}_{ssp}_{realization}_{grid}_{date}.nc'
        
        with xr.open_dataset(nc_name) as nc:
            array = nc[var].values
            chunk_days = array.shape[2]  # Get the number of days in this chunk

            for p, (lat, lon) in enumerate(zip(guide['lat'], guide['lon'])):
                Y = int(((lat + 90) / lat_res) + 1)
                X = int((lon / lon_res) + 1)
                pixels[start_index:start_index+chunk_days, p] = array[X, Y, :]

        start_index += chunk_days

    return pixels

def create_netcdf(data, var, ssp, days):
    output_dir = f'{base_path}/cmip6/cmip6_subset/{model}/{ssp}/{var}'
    os.makedirs(output_dir, exist_ok=True)
    nc_name = f'{output_dir}/{var}_day_{model}_{realization}_{ssp}_subset.nc'

    dim_name = variables.iloc[var-1, 2]
    dim_long_name = variables.iloc[var-1, 4]
    dim_units = variables.iloc[var-1, 6]

    LON_n = len(guide['lon'].unique())
    LAT_n = len(guide['lat'].unique())
    TIME_n = len(days)

    with Dataset(nc_name, 'w', format='NETCDF4') as nc_out:
        nc_out.createDimension('lon', LON_n)
        nc_out.createDimension('lat', LAT_n)
        nc_out.createDimension('time', TIME_n)

        lons = nc_out.createVariable('lon', 'f4', ('lon',))
        lats = nc_out.createVariable('lat', 'f4', ('lat',))
        times = nc_out.createVariable('time', 'i4', ('time',))
        var_out = nc_out.createVariable(dim_name, 'f8', ('lon', 'lat', 'time'), fill_value=-9999)

        lons.units = 'degrees_east'
        lons.long_name = 'Longitude'
        lats.units = 'degrees_north'
        lats.long_name = 'Latitude'
        times.units = 'days since 1981-01-01' if ssp == 'historical' else 'days since 2015-01-01'
        times.long_name = 'days since 19810101' if ssp == 'historical' else 'days since 20150101'
        var_out.units = dim_units
        var_out.long_name = dim_long_name

        lons[:] = guide['lon'].unique()
        lats[:] = guide['lat'].unique()
        times[:] = days
        var_out[:, :, :] = data.reshape(LON_n, LAT_n, TIME_n)

    print(f"Created {nc_name}")

for s, ssp in enumerate(ssps):
    print(f"Processing {ssp}")
    
    dates = ["19810101-19851231", "19860101-19901231", "19910101-19951231", "19960101-20001231", "20010101-20051231", "20060101-20101231", "20110101-20141231"] if ssp == "historical" else ["20150101-20151231", "20160101-20201231", "20210101-20251231", "20260101-20301231", "20310101-20351231", "20360101-20401231", "20410101-20451231", "20460101-20501231", "20510101-20551231", "20560101-20601231", "20610101-20651231", "20660101-20701231", "20710101-20751231", "20760101-20801231", "20810101-20851231", "20860101-20901231", "20910101-20951231", "20960101-21001231"]
    
    for v in range(1, 2):
        var = variables.iloc[v-1, 2]
        print(f"Processing {var}")
        
        if ssp == "historical":
            data = process_historical(var, ssp, dates)
            create_netcdf(data, v, ssp, historical_days)
        else:
            data = process_future(var, ssp, dates)
            create_netcdf(data, v, ssp, ssp_days)

print("Processing complete.")