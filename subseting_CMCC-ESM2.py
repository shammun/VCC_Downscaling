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
base_path = '/content/drive/My Drive/VCCr'  # Replace with your actual folder path

# Read input files
future_dates = pd.read_csv(f'{base_path}/future_dates_2023.csv')
historical_dates = pd.read_csv(f'{base_path}/historical_dates_2023.csv')
models = pd.read_csv(f'{base_path}/models_2023.csv')
variables = pd.read_csv(f'{base_path}/Variables.csv')
ssps = ["historical", "ssp245", "ssp370", "ssp585"]

historical_days = np.arange(1, 12776)
ssp_days = np.arange(12776, 44166)

# Model specific information
m = 1  # Model index
model = models.iloc[m-1, 0]
realization = models.iloc[m-1, 3]
grid = models.iloc[m-1, 4]
guide = gpd.read_file(f'{base_path}/grids/subsets/{model}_guide.shp')
lon_res = models.iloc[m-1, 24]  # pixel resolutions
lat_res = models.iloc[m-1, 23]

def process_historical(var, ssp, dates):
    pixels1 = np.full((7300, len(guide)), -999999, dtype=np.float32)
    pixels2 = np.full((5475, len(guide)), -999999, dtype=np.float32)

    for d in range(7):
    #for d in range(1, 3):
        date = dates[d-1]
        print(f"Processing {date}")
        nc_name = f'{base_path}/cmip6/{var}/{model}/{ssp}/{var}_day_{model}_{ssp}_{realization}_{date}.nc'
        #nc_name = f'{base_path}/cmip6/{var}/{model}/{ssp}/{var}_day_{model}_{ssp}_{realization}_{grid}_{date}.nc'
        
        with xr.open_dataset(nc_name) as nc:
            array = nc[var].values

            for p, (lat, lon) in enumerate(zip(guide['lat'], guide['lon'])):
                Y = int(((lat + 90) / lat_res) + 1)
                X = int((lon / lon_res) + 1)

                if d == 1:
                    pixels1[:, p] = array[X, Y, 1825:9125]
                else:
                    pixels2[:, p] = array[X, Y, :5475]

    data = np.vstack((pixels1, pixels2))
    return data

def process_future(var, ssp, dates):
    pixels = [np.full((9125, len(guide)), -999999, dtype=np.float32) for _ in range(3)]
    pixels.append(np.full((4015, len(guide)), -999999, dtype=np.float32))

    for d in range(1, 5):
        date = dates[d-1]
        print(f"Processing {date}")
        nc_name = f'{base_path}/cmip6/{var}/{model}/{ssp}/{var}_day_{model}_{ssp}_{realization}_{grid}_{date}.nc'
        
        with xr.open_dataset(nc_name) as nc:
            array = nc[var].values

            for p, (lat, lon) in enumerate(zip(guide['lat'], guide['lon'])):
                Y = int(((lat + 90) / lat_res) + 1)
                X = int((lon / lon_res) + 1)

                if d < 4:
                    pixels[d-1][:, p] = array[X, Y, :9125]
                else:
                    pixels[d-1][:, p] = array[X, Y, :4015]

    data = np.vstack(pixels)
    return data

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
        times.units = 'days since 1980-01-01'
        times.long_name = 'days since 19800101'
        var_out.units = dim_units
        var_out.long_name = dim_long_name

        lons[:] = guide['lon'].unique()
        lats[:] = guide['lat'].unique()
        times[:] = days
        var_out[:, :, :] = data.reshape(LON_n, LAT_n, TIME_n)

    print(f"Created {nc_name}")

for s, ssp in enumerate(ssps, 1):
    print(f"Processing {ssp}")
    
    dates_n = models.iloc[m-1, 5] if ssp == "historical" else models.iloc[m-1, 6]
    dates = historical_dates.iloc[:dates_n, m-1] if ssp == "historical" else future_dates.iloc[:dates_n, m-1]
    
    for v in range(1, 2):
    #for v in range(1, 4):
        var = variables.iloc[v-1, 2]
        print(f"Processing {var}")
        
        if ssp == "historical":
            data = process_historical(var, ssp, dates)
            create_netcdf(data, v, ssp, historical_days)
        else:
            data = process_future(var, ssp, dates)
            create_netcdf(data, v, ssp, ssp_days)

print("Processing complete.")
