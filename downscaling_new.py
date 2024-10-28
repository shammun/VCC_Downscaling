import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

def calculate_climatological_diurnal_cycle(era5_data):
    """Calculate the climatological diurnal cycle <f_E>(i,j,h,d)"""
    daily_total = era5_data.resample(time='1D').sum()
    hourly_fraction = xr.where(daily_total > 0, era5_data.groupby('time.hour') / daily_total, 0)
    climatology = hourly_fraction.groupby('time.dayofyear').mean('time')
    return climatology

def regrid_diurnal_cycle(diurnal_cycle, target_lon, target_lat):
    """Regridding function R: <f_G>(i',j',h,d) ≡ R[<f_E>(i,j,h,d)] using bi-linear interpolation"""
    source_lon = diurnal_cycle.longitude.values
    source_lat = diurnal_cycle.latitude.values
    
    regridded_data = np.zeros((len(target_lat), len(target_lon), diurnal_cycle.dayofyear.size, diurnal_cycle.hour.size))
    
    for d in range(diurnal_cycle.dayofyear.size):
        for h in range(diurnal_cycle.hour.size):
            interpolator = RegularGridInterpolator(
                (source_lat, source_lon),
                diurnal_cycle.isel(dayofyear=d, hour=h).values,
                method='linear'  # This specifies bi-linear interpolation
            )
            
            XI, YI = np.meshgrid(target_lon, target_lat)
            regridded_data[:, :, d, h] = interpolator((YI, XI))
    
    return xr.DataArray(
        regridded_data,
        coords={
            'latitude': target_lat,
            'longitude': target_lon,
            'dayofyear': diurnal_cycle.dayofyear,
            'hour': diurnal_cycle.hour
        },
        dims=['latitude', 'longitude', 'dayofyear', 'hour']
    )

def create_synthetic_diurnal_cycle(regridded_diurnal_cycle, daily_data):
    """P̃_G0(i',j',h,d,y) ≡ <f_G>(i',j',h,y) * P_G(i',j',d,y)"""
    return regridded_diurnal_cycle * daily_data

def conserve_daily_total(synthetic_diurnal_cycle, daily_data):
    """P̃_G(i',j',h,d,y) ≡ P̃_G0(i',j',h,d,y) * (P_N(i',j',d,y) / Σ_h=1^24 P̃_G0(i',j',h,d,y))"""
    daily_sum = synthetic_diurnal_cycle.sum(dim='hour')
    correction_factor = daily_data / daily_sum
    return synthetic_diurnal_cycle * correction_factor

def downscale_precipitation(era5_data, daily_data):
    """Main function to downscale precipitation data using bi-linear interpolation."""
    # Step 1: Calculate climatological diurnal cycle <f_E>(i,j,h,d)
    diurnal_cycle = calculate_climatological_diurnal_cycle(era5_data)
    
    # Step 2: Regrid to fine-scale target grid <f_G>(i',j',h,d) ≡ R[<f_E>(i,j,h,d)] using bi-linear interpolation
    regridded_diurnal_cycle = regrid_diurnal_cycle(
        diurnal_cycle, 
        daily_data.longitude.values, 
        daily_data.latitude.values
    )
    
    # Step 3: Create synthetic diurnal cycle P̃_G0(i',j',h,d,y)
    synthetic_diurnal_cycle = create_synthetic_diurnal_cycle(regridded_diurnal_cycle, daily_data)
    
    # Step 4: Conserve daily total P̃_G(i',j',h,d,y)
    downscaled_data = conserve_daily_total(synthetic_diurnal_cycle, daily_data)
    
    return downscaled_data

# Example usage:
# era5_data = xr.open_dataarray('path_to_era5_data.nc')
# nclimgrid_data = xr.open_dataarray('path_to_nclimgrid_data.nc')
# downscaled_data = downscale_precipitation(era5_data, nclimgrid_data)