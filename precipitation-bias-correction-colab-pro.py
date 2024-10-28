import os
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from google.colab import drive
from multiprocessing import Pool, cpu_count
from dbfread import DBF  # New import for reading .dbf files

# Mount Google Drive
drive.mount('/content/drive')

# Set working directory
os.chdir("/content/drive/MyDrive/VCC/")
print(os.getcwd())

# Load guides (adjust paths as necessary)
models = pd.read_csv("models_2023.csv")
variables = pd.read_csv("Variables.csv")
ssps = ["historical", "ssp245", "ssp370", "ssp585"]
vars = ["pr", "tasmax", "tasmin"]

def read_dbf(dbf_path):
    """Read a .dbf file and return a pandas DataFrame."""
    return pd.DataFrame(iter(DBF(dbf_path)))

def process_pixel(args):
    i, guide, lon, lat, lon_res, lat_res, lat_length, prism_array, hist_array, to_bc_array, ssp = args
    
    guide_lon = guide.iloc[i, 4]  # get central coordinates (lon flip)
    guide_lat = guide.iloc[i, 2]  # get central coordinates
    
    X = round((guide_lon - lon[0]) / lon_res) + 1
    Y = int(round(lat_length - abs((guide_lat - lat[-1]) / lat_res)))
    
    prism_vector = prism_array[X-1, Y-1, :12410]  # 1981 - 2014
    
    hist_vector = hist_array[X-1, Y-1, 365:12775]  # 1981 - 2014
    hist_vector = np.round(hist_vector * 86400, 1)
    
    if np.isnan(prism_vector[0]):
        return np.full(31390 if ssp != "historical" else 12410, -9999)
    
    to_bc_vector = to_bc_array[X-1, Y-1, 365:12775] if ssp == "historical" else to_bc_array[X-1, Y-1, :31390]
    to_bc_vector = np.round(to_bc_vector * 86400, 1)
    
    prism_cdf = stats.ecdf(prism_vector)
    hist_cdf = stats.ecdf(hist_vector)
    
    probability = hist_cdf(to_bc_vector)
    pixel = np.quantile(prism_vector, probability)
    
    return pixel

def process_chunk(chunk_data):
    chunk_start, chunk_end, guide, lon, lat, lon_res, lat_res, lat_length, prism_array, hist_array, to_bc_array, ssp = chunk_data
    
    chunk_results = []
    for i in range(chunk_start, chunk_end):
        result = process_pixel((i, guide, lon, lat, lon_res, lat_res, lat_length, prism_array, hist_array, to_bc_array, ssp))
        chunk_results.append(result)
    
    return chunk_results

def main():
    chunk_size = 500  # To utilize Colab Pro's higher RAM
    
    for v in range(1):  # only pr
        var = variables.iloc[v, 3]
        print(f"Processing variable: {var}")
        
        for m in [1, 11]:
        #for m in range(17):
            model = models.iloc[m-1, 0]
            print(f"Processing model: {model}")
            guide = read_dbf(f"grids/subsets/{model}_guide.dbf")  # Changed to read .dbf file
            realization = models.iloc[m, 4]
            
            with xr.open_dataset(f"prism/resampled/prism_{var}_day_{model}_resampled.nc") as nc1:
                lon = nc1.lon.values
                lon_res = abs(lon[1] - lon[0])
                lat = nc1.lat.values
                lat_res = abs(lat[1] - lat[0])
                lat_length = len(lat)
                prism_array = nc1[var].values
            
            with xr.open_dataset(f"cmip6/cmip6_subset/{model}/historical/{var}/{var}_day_{model}_{realization}_historical_subset.nc") as nc_hist:
                hist_array = nc_hist[var].values
            
            for s in range(1):
                ssp = ssps[s]
                print(f"Processing scenario: {ssp}")
                
                with xr.open_dataset(f"cmip6/cmip6_subset/{model}/{ssp}/{var}/{var}_day_{model}_{realization}_{ssp}_subset.nc") as nc:
                    to_bc_array = nc[var].values
                
                all_results = []
                
                total_chunks = len(guide) // chunk_size + (1 if len(guide) % chunk_size != 0 else 0)
                
                for chunk_index, chunk_start in enumerate(range(0, len(guide), chunk_size)):
                    chunk_end = min(chunk_start + chunk_size, len(guide))
                    chunk_data = (chunk_start, chunk_end, guide, lon, lat, lon_res, lat_res, lat_length, prism_array, hist_array, to_bc_array, ssp)
                    
                    with Pool(cpu_count()) as pool:
                        chunk_results = pool.map(process_chunk, [chunk_data])
                    
                    all_results.extend(chunk_results[0])
                    
                    print(f"Processed chunk {chunk_index + 1}/{total_chunks} ({chunk_start} to {chunk_end})")
                    
                    # Save intermediate results every 5 chunks
                    if (chunk_index + 1) % 5 == 0 or chunk_index == total_chunks - 1:
                        intermediate_data_array = np.array(all_results).T
                        intermediate_ds = xr.Dataset(
                            {var: (["lon", "lat", "time"], intermediate_data_array)},
                            coords={
                                "lon": ("lon", guide['lon_flip'].unique()),
                                "lat": ("lat", guide['lat'].unique()),
                                "time": ("time", np.arange(1, intermediate_data_array.shape[2] + 1))
                            }
                        )
                        intermediate_ds[var].attrs["units"] = variables.iloc[v, 7]
                        intermediate_ds[var].attrs["long_name"] = variables.iloc[v, 5]
                        intermediate_ds.to_netcdf(f"output/intermediate_{ssp}_{var}_day_{model}_BC_chunk{chunk_index + 1}.nc")
                        print(f"Saved intermediate results up to chunk {chunk_index + 1}")
                
                data_array = np.array(all_results).T
                
                # Create final xarray Dataset
                ds = xr.Dataset(
                    {var: (["lon", "lat", "time"], data_array)},
                    coords={
                        "lon": ("lon", guide['lon_flip'].unique()),
                        "lat": ("lat", guide['lat'].unique()),
                        "time": ("time", np.arange(1, data_array.shape[2] + 1))
                    }
                )
                
                # Set attributes
                ds[var].attrs["units"] = variables.iloc[v, 7]
                ds[var].attrs["long_name"] = variables.iloc[v, 5]
                
                # Save final NetCDF
                ds.to_netcdf(f"output/{ssp}_{var}_day_{model}_BC.nc")
                print(f"Saved final results for {ssp} {var} {model}")

if __name__ == "__main__":
    main()
