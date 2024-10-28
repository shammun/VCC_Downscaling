import os
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import multiprocessing as mp
from dbfread import DBF
import time
from google.colab import drive
from tqdm import tqdm

# Mount Google Drive
drive.mount('/content/drive')

# Set the base directory in Google Drive
base_dir = '/content/drive/My Drive/VCC'  # Adjust this path as needed

# Load necessary files to guide the loops
models = pd.read_csv(os.path.join(base_dir, 'models_2023.csv'))
variables = pd.read_csv(os.path.join(base_dir, 'Variables.csv'))

# Function to process pixels (equivalent to the inner foreach loop in R)
def process_pixel(args):
    X_, Y_, arrays = args
    pixel = []
    for array in arrays:
        pixel.extend(array[X_, Y_, :1825])
    return pixel

# Function to process a single variable and model
def process_var_model(var, model, arrays, guide, lon, lat, lon_res, lat_res):
    print(f"Processing {var} for model {model}")
    
    guide_lat_res = models.loc[models.iloc[:, 0] == model, 23].values[0]
    guide_lon_res = models.loc[models.iloc[:, 0] == model, 24].values[0]
    lat_length = len(lat)
    
    corse_pixel = np.full(14600, -9999)
    
    for p in tqdm(range(len(guide)), desc=f"{var} - {model}"):
        guide_lon = guide.records[p]['lon_flip']
        guide_lat = guide.records[p]['lat']
        
        X = round((guide_lon - lon[0]) / lon_res) + 1
        Y = int(round(lat_length - (guide_lat - lat[-1])/lat_res))
        
        X_list = np.arange(guide_lon - 0.5 * guide_lon_res, guide_lon + 0.5 * guide_lon_res + lon_res, lon_res)
        Y_list = np.arange(guide_lat - 0.5 * guide_lat_res, guide_lat + 0.5 * guide_lat_res + lat_res, lat_res)
        
        X_Y_list = [(round((x - lon[0]) / lon_res) + 1,
                     int(round(lat_length - (y - lat[-1])/lat_res)))
                    for x, y in np.array(np.meshgrid(X_list, Y_list)).T.reshape(-1, 2)]
        
        # Process pixels
        pixels = [process_pixel((X_, Y_, arrays)) for X_, Y_ in X_Y_list]
        
        # Average values
        pixel = np.mean(pixels, axis=0)
        corse_pixel = np.column_stack((corse_pixel, pixel))
    
    return corse_pixel

# Main processing function
def main():
    # Set up multiprocessing
    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=num_cores)
    
    for v in range(len(variables)):
        var = variables.iloc[v, 2]
        print(f"Processing variable: {var}")
        
        # Load all ncs per variable
        ncs = [Dataset(os.path.join(base_dir, f"prism/prism_{var}_day_{year}0101-{year}1231.nc"))
               for year in range(1981, 2021, 5)]
        
        # Extract lat, lon dimensions and resolution from ncs
        lon = np.unique(ncs[0].variables['lon'][:])
        lon_res = abs(lon[1] - lon[0])
        lat = np.unique(ncs[0].variables['lat'][:])
        lat_res = abs(lat[1] - lat[0])
        
        # Create arrays
        arrays = [nc.variables[var][:] for nc in ncs]
        
        # Close ncs
        for nc in ncs:
            nc.close()
        
        # Process only the 1st and 11th models
        selected_models = [1, 11]
        model_results = []
        
        for m in selected_models:
            model = models.iloc[m-1, 0]  # Subtract 1 because Python uses 0-based indexing
            guide = DBF(os.path.join(base_dir, f"grids/subsets/{model}_guide.dbf"))
            model_results.append(pool.apply_async(process_var_model, (var, model, arrays, guide, lon, lat, lon_res, lat_res)))
        
        # Collect results
        for i, result in enumerate(model_results):
            m = selected_models[i]
            model = models.iloc[m-1, 0]  # Subtract 1 because Python uses 0-based indexing
            corse_pixel = result.get()
            
            print(f"Creating NetCDF for {var} - {model}")
            
            # Organize data to fit in array
            data = corse_pixel[:, 1:]
            data = data.T
            
            # Define dimensions for the array
            guide = DBF(os.path.join(base_dir, f"grids/subsets/{model}_guide.dbf"))
            LON_n = len(np.unique([rec['lon_flip'] for rec in guide]))
            LAT_n = len(np.unique([rec['lat'] for rec in guide]))
            TIME_n = 14600
            
            # Create the Array
            data_array = data.reshape((LON_n, LAT_n, TIME_n))
            
            # Create NetCDF file
            nc_name = os.path.join(base_dir, 'resampled', f"prism_{var}_day_{model}_resampled.nc")
            nc_out = Dataset(nc_name, 'w', format='NETCDF4')
            
            # Define dimensions
            nc_out.createDimension('lon', LON_n)
            nc_out.createDimension('lat', LAT_n)
            nc_out.createDimension('time', TIME_n)
            
            # Create variables
            lon_var = nc_out.createVariable('lon', 'f4', ('lon',))
            lat_var = nc_out.createVariable('lat', 'f4', ('lat',))
            time_var = nc_out.createVariable('time', 'i4', ('time',))
            data_var = nc_out.createVariable(var, 'f8', ('lon', 'lat', 'time',), 
                                             zlib=True, complevel=9, fill_value=-9999)
            
            # Add data
            lon_var[:] = np.unique([rec['lon_flip'] for rec in guide])
            lat_var[:] = np.unique([rec['lat'] for rec in guide])
            time_var[:] = np.arange(1, 14601)
            data_var[:] = data_array
            
            # Add attributes
            lon_var.units = 'degrees_east'
            lon_var.long_name = 'Longitude'
            lat_var.units = 'degrees_north'
            lat_var.long_name = 'Latitude'
            time_var.units = 'days'
            time_var.long_name = 'days since 19810101'
            data_var.units = variables.iloc[v, 6]
            data_var.long_name = variables.iloc[v, 4]
            
            # Add global attributes
            nc_out.Name = f"prism NetCDF resampled to {model} model resolution"
            nc_out.Version = "NA"
            nc_out.Author = "Andre Geraldo de Lima Moraes"
            nc_out.Institution = "Utah State University, Watershed Sciences Department"
            nc_out.Address = "5210 Old Main Hill, NR 210, Logan, UT 84322"
            nc_out.email = "andre.moraes@usu.edu"
            nc_out.Description = f"This is the same as the prism data, but in NetCDF format and resampled to {model} model resolution"
            nc_out.lineage = "Parameter-elevation Relationships on Independent The Slopes Model (PRISM 4K) project (https://prism.oregonstate.edu/)"
            nc_out.License = "Same as PRISM"
            nc_out.fees = "This data set is free"
            nc_out.Disclaimer = "While every effort has been made to ensure the accuracy and completeness of the data, no guarantee is given that the information provided is error-free or that the dataset will be suitable for any particular purpose. Users are advised to use this dataset with caution and to independently verify the data before making any decisions based on it. The creators of this dataset make no warranties, express or implied, regarding the dataset's accuracy, reliability, or fitness for a particular purpose. In no event shall the creators be liable for any damages, including but not limited to direct, indirect, incidental, special, or consequential damages, arising out of the use or inability to use the dataset. Users of this dataset are encouraged to properly cite the dataset in any publications or works that make use of the data. By using this dataset, you agree to these terms and conditions. If you do not agree with these terms, please do not use the dataset."
            
            # Close NetCDF
            nc_out.close()
    
    # Close the multiprocessing pool
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
    print("Finished")
