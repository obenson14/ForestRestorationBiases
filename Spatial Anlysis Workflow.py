# SPATIAL ANALYSIS WORKFLOW TO RUN IN JUYPTER NOTEBOOKS USING GEE PYTHON API (WITH PLACEHOLDER DATASETS AND PROJECT ID

# Import Libraries
import ee
import pandas as pd
import numpy as np
import requests
from rasterio.io import MemoryFile
import pylandstats as pls
from tqdm import tqdm
import warnings

# Supress runtime warnings for a cleaner output
warnings.filterwarnings("ignore")

# Initialise and access my Google Earth Engine Project and session
ee.Authenticate()
ee.Initialize(project='PROJECT ID')

# Load pre-processed dataset as .csv (from local storage)
df = pd.read_csv("INPUT_DATASET")

# Remapping the detailed GLC_FCS30D 35 landuse classes to 10 simplified ones
# Full list of classes available at - https://zenodo.org/records/8239305
landuse_mapping = {
    10: 1, 11: 1, 12: 1, 20: 1,
    51: 2, 52: 2, 61: 2, 62: 2, 71: 2, 72: 2, 81: 2, 82: 2, 91: 2, 92: 2,
    120: 3, 121: 3, 122: 3,
    130: 4,
    150: 5, 152: 5, 153: 5,
    181: 6, 182: 6, 183: 6, 184: 6, 185: 6, 186: 6, 187: 6,
    190: 7,
    200: 8, 201: 8, 202: 8,
    210: 9,
    220: 10
}

# Defining the names of each of these classes
basic_class_names = {
    1: "Arable land/Crops",
    2: "Forest",
    3: "Shrubland",
    4: "Grassland/Pasture",
    5: "Sparse Vegetation",
    6: "Wetlands",
    7: "Impervious Surfaces (Urban)",
    8: "Barren",
    9: "Water",
    10: "Snow/Ice"
}

# Function to generate a classified GLC image for a given buffer and band
def classify_glc(glc_collection, glc_band_name, buffer):
    '''
    Generates a landcover raster with remapped classes and clipped to the site location and buffer:
    
    Steps:
    - Filters GLC_FCS30D image with the buffer boundary
    - Selects the necessary 'band' (representing different years) for the study year
    - In scenarios where the buffer is on the boundary of a GLC tile, a mosiac is created
    - Remapping of the 35 numeric class values to simplified named values
    - Clips to study buffer

    Args:
    - glc_collection - GLC image collection filtered for date
    - glc_band_name - Band name representing year
    - buffer - Earth Engine geometry representing the study site buffer

    Returns:
    - A classified land cover scene from the GLC dataset for the study year (or closest year to), 
    rempped and clipped to study site
    '''
    return (
        glc_collection.filterBounds(buffer).mosaic()
        .select(glc_band_name)
        .remap(list(landuse_mapping.keys()), list(landuse_mapping.values()))
        .rename("classification")
        .clip(buffer)
    )

# Create list to store results in for output dataframe
all_results = []

# MAIN LOOP FOR ANALYSIS

# Loop through each study site with tqdm for progress bar in output (for Jupyter environments)
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing sites"):
    try:
        # Extract study site info
        site_id = int(row['ID'])
        lon = float(row['Longitude'])
        lat = float(row['Latitude'])
        year = int(row['Study Year'])

        # Define GEE point geometry for study site location
        study_site = ee.Geometry.Point([lon, lat])

        # Define study area (10km buffer)
        buffer_10km = study_site.buffer(10000)  

        # GLC DATA COLLECTION
        # Collection code source - https://gee-community-catalog.org/projects/glc_fcs/

        if year >= 2000: # Select annual datasets from year 2000 onwards
            glc_collection = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual")
            band_index = year - 1999
            glc_band_name = f"b{band_index}"
            training_id = f"GLC Annual {year}"
        else:
            if year >= 1997: # Between years 1997 - 1999, Select year 2000 annual data as the closest year
                glc_collection = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual")
                glc_band_name = "b1"
                training_id = "GLC Annual 2000"
            else:
                five_years = [1985, 1990, 1995] # For earlier years, use the closest five-year interval dataset
                closest_year = min(five_years, key=lambda y: abs(year - y))
                band_map = {1985: 'b1', 1990: 'b2', 1995: 'b3'}
                glc_band_name = band_map[closest_year]
                glc_collection = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/five-years-map")
                training_id = f"GLC Five-Year {closest_year}"

        # Use 'classify_glc' function to create clipped and remapped GLC raster for the study site
        classified = classify_glc(glc_collection, glc_band_name, buffer_10km)

        # Create dictionary for new buffers sizes for spatial analysis statistics
        stats_buffers = {
            '1km': 1000,
            '2.5km': 2500,
            '5km': 5000,
            '10km': 10000
        }

        # Loop through the differet buffer sizes 
        for buffer_label, buffer_radius in stats_buffers.items():
            # Create buffer geometry from new buffer sizes for each location
            stats_buffer = study_site.buffer(buffer_radius)

            # Calculate land use class distribution histogram in buffer area
            # Using Earth Engine's frequencyHistogram reducer for pixel value count
            # https://developers.google.com/earth-engine/tutorials/community/introduction-to-dynamic-world-pt-1
            histogram = classified.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=stats_buffer,
                scale=30,
                maxPixels=1e9,
                bestEffort=True
            ).get('classification').getInfo() or {}

            # Download classified image for landscape metrics in local memory in order to use 'pylandstats' 
            url = classified.getDownloadURL({
                'scale': 30,
                'region': stats_buffer.bounds().getInfo(),
                'maxPixels': 1e9,
                'filePerBand': False,
                'format': 'GeoTIFF'
            })
            response = requests.get(url)

            # Read downloaded GeoTIFF directly into memory using rasterio.MemoryFile
            # to writing temporary files to disk and so improves efficiency.
            # Documentation - https://rasterio.readthedocs.io/en/stable/topics/memory-files.html

            # Loop for retries added for scenarios where rasterio fails to read the GeoTIFF correctly
            # resulting in failed site. Tries up to 3 times before moving on
            for attempt in range(3):
                try:
                    response = requests.get(url)
                    with MemoryFile(response.content) as memfile:
                        with memfile.open() as dataset:
                            classified_np = dataset.read(1)
                    break  # Success = break loop
                except Exception:
                    # after 3rd failure move on
                    if attempt == 2:
                        raise  

            pixel_size = 30
            pixel_area = pixel_size * pixel_size

            # Calculate landscape metrics with pylandstats library
            # Metrics used include; patch density, edge density, mean patch area, largest patch index, contagion
            # Documentation at - https://pylandstats.readthedocs.io/en/latest/landscape.html 
            landscape = pls.Landscape(classified_np, res=(pixel_size, pixel_size))
            metrics_df = landscape.compute_class_metrics_df()
            contagion_val = landscape.contagion() 
                    
            # Extract forest metrics (class 2)
            forest_class = 2
            if forest_class in metrics_df.index:
                forest_metrics = metrics_df.loc[forest_class]
                pd_val = forest_metrics['patch_density']
                ed_val = forest_metrics['edge_density']
                # Convert mean patch area from pixels to hectares (1 ha = 10,000 m^2)
                mpa_val = (forest_metrics['area_mn'] * pixel_area) / 10000 
                lpi_val = forest_metrics['largest_patch_index']
            else:
                # If metrics aren't calculable, default to 0.0 
                pd_val = ed_val = mpa_val = lpi_val = 0.0

            # Add results for current site metadata and buffer metrics to dictionary
            summary = {
                'ID': site_id,
                'Stats Buffer': buffer_label,
                'GLC Image ID': training_id,
                'GLC Image Band': glc_band_name,
                'Forest Patch Density': pd_val,
                'Forest Edge Density': ed_val,
                'Forest Mean Patch Area (ha)': mpa_val,
                'Forest Largest Patch Index': lpi_val,
                'Contagion': round(contagion_val, 2)
            }

            # Calculate percentage cover of each land use class from histogram counts
            # https://developers.google.com/earth-engine/tutorials/community/introduction-to-dynamic-world-pt-2
            if histogram:
                total_pixels = sum(histogram.values())
                for class_str, count in histogram.items():
                    class_id = int(class_str)
                    percent_cover = (count / total_pixels) * 100
                    landuse_name = basic_class_names[class_id]
                    summary[f'{landuse_name} %'] = percent_cover

            # Ensure all landuse classes are represented in output (0% if not present)
            for class_id, name in basic_class_names.items():
                key = f"{name} %"
                if key not in summary:
                    summary[key] = 0.0

            # Round all float values to 2 decimal places
            for key, value in summary.items():
                if isinstance(value, float):
                    summary[key] = round(value, 2)

            # Append the summary dictionary to results list
            all_results.append(summary)

     # Print error message for any failed iteration but continue processing next sites
    except Exception as e:
        print(f"Failed on ID {row['ID']}: {str(e)}")
        continue

# Convert list of result dictionaries to pandas DataFrame for final output
result_df = pd.DataFrame(all_results)

# Save dataframe as csv to local file
result_df.to_csv("RESULTS DATASET", index=False)
