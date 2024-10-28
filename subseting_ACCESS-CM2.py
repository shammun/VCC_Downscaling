import os
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import geopandas as gpd
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set the base path for your project in Google Drive
base_path = '/content/drive/My Drive/VCC'  # Replace with your actual folder path

# Read input files
future_dates = pd.read_csv(f"{base_path}/future_dates_2023.csv")
historical_dates = pd.read_csv(f"{base_path}/historical_dates_2023.csv")
models = pd.read_csv(f"{base_path}/models_2023.csv")
variables = pd.read_csv(f"{base_path}/Variables.csv")
ssps = ["historical", "ssp245", "ssp370", "ssp585"]

historical_days = np.arange(1, 12411)
ssp_days = np.arange(12776, 44166)

# Model specific information
m = 11  # Model index
model = models.iloc[m-1, 0]
realization = models.iloc[m-1, 3]
grid = models.iloc[m-1, 4]
guide = gpd.read_file(f"{base_path}/grids/subsets/{model}_guide.shp")
lon_res = 360 / 192  # CHANGES FOR EVERY MODEL
lat_res = 180 / 144

def process_file(args):
    var, ssp, date, guide, lon_res, lat_res = args
    nc_name = f'{base_path}/cmip6/{var}/{model}/{ssp}/{var}_day_{model}_{ssp}_{realization}_{date}.nc'
    #nc_name = f'{base_path}/cmip6/{var}/{model}/{ssp}/{var}_day_{model}_{ssp}_{realization}_{grid}_{date}.nc'
        
    with xr.open_dataset(nc_name, chunks={'time': 1000}) as nc:
        array = nc[var].values
        
        pixels = np.full((array.shape[2], len(guide)), -999999, dtype=np.float32)
        
        for p, (lat, lon) in enumerate(zip(guide['lat'], guide['lon'])):
            Y = int(((lat + 90) / lat_res) + 1)
            X = int((lon / lon_res) + 1)
            
            pixel = array[X, Y, :]
            
            for r in range(12):
                remove = (r * 1460) + 1
                pixel = np.delete(pixel, remove)
            
            pixels[:, p] = pixel
    
    return pixels

def process_scenario(ssp, dates, days):
    for v in range(3):
        var = variables.iloc[v, 2]
        print(f"Processing {var} for {ssp}")
        
        # Prepare arguments for multiprocessing
        args_list = [(var, ssp, date, guide, lon_res, lat_res) for date in dates]
        
        # Use multiprocessing to process files in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            pixels_list = list(tqdm(pool.imap(process_file, args_list), total=len(args_list)))
        
        data = np.vstack(pixels_list)
        
        # Creating the netcdf
        output_dir = f'{base_path}/cmip6/cmip6_subset/{model}/{ssp}/{var}'
        os.makedirs(output_dir, exist_ok=True)
        nc_name = f'{output_dir}/{var}_day_{model}_{realization}_{ssp}_subset.nc'
        
        dim_name = variables.iloc[v, 2]
        dim_long_name = variables.iloc[v, 4]
        dim_units = variables.iloc[v, 6]
        
        LON_n = len(guide['lon'].unique())
        LAT_n = len(guide['lat'].unique())
        TIME_n = len(days)
        
        with Dataset(nc_name, 'w', format='NETCDF4') as nc_out:
            # Define dimensions
            nc_out.createDimension('lon', LON_n)
            nc_out.createDimension('lat', LAT_n)
            nc_out.createDimension('time', TIME_n)
            
            # Define variables
            lons = nc_out.createVariable('lon', 'f4', ('lon',))
            lats = nc_out.createVariable('lat', 'f4', ('lat',))
            times = nc_out.createVariable('time', 'i4', ('time',))
            var_out = nc_out.createVariable(dim_name, 'f4', ('lon', 'lat', 'time'), fill_value=-9999, chunksizes=(LON_n, LAT_n, 1000))
            
            # Add attributes
            lons.units = 'degrees_east'
            lons.long_name = 'Longitude'
            lats.units = 'degrees_north'
            lats.long_name = 'Latitude'
            times.units = 'days since 1980-01-01'
            times.long_name = 'days since 19800101'
            var_out.units = dim_units
            var_out.long_name = dim_long_name
            
            # Write data
            lons[:] = guide['lon'].unique()
            lats[:] = guide['lat'].unique()
            times[:] = days
            var_out[:, :, :] = data.reshape(LON_n, LAT_n, TIME_n)
        
        print(f"Created {nc_name}")

# Process historical scenario
dates_historical = historical_dates.iloc[:models.iloc[m-1, 5], m-1]
process_scenario('historical', dates_historical, historical_days)

# Process future scenarios
for ssp in ['ssp245', 'ssp370', 'ssp585']:
    dates_future = future_dates.iloc[:models.iloc[m-1, 6], m-1]
    process_scenario(ssp, dates_future, ssp_days)

print("Processing complete.")
