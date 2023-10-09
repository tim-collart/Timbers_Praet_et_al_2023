# In-situ sensor data and Python scripts developed for the TIMBERS project.

This repository contains in-situ sensor data and the python code that were used in the research article: Praet, N.; Collart, T.; Ollevier, A.; Roche, M.; Degrendele, K.; De Rijcke, M.; Urban, P.; Vandorpe, T. The Potential of Multibeam Sonars as 3D Turbidity and SPM Monitoring Tool in the North Sea. *Remote Sens.* **2023**

It contains following ipython notebooks:

## 0. Pre-process the water column data: [0_mbes_preprocessing.ipynb](./0_mbes_preprocessing.ipynb)
This notebook executes following steps:
* Read ASCII files exported from Sonarscope 
* Pre-process the data:
    * Format columns to to match Entwine header
    * Remove points with missing values
    * Crop data points to Belgian EEZ to remove spatial outliers
    * Calculate RGBA values for visualization of data in Potree viewer
* Export to CSV files for ingestion into Entwine

## 1. Ingest CSV files in entwine: [1_mbes_entwine_ingest.ipynb](./1_mbes_entwine_ingest.ipynb)
This notebook was run for each dataset (defined by the input path) and executes following steps:
* Ingest the pre-processed water column data files into entwine point tiles (EPT) which are octtree data structures that allow for fast location querying of the data.
* Optionally visualizes the ingested dataset in Potree Viewer.

## 2. Pre-process the in-situ data: [2_lisst_preprocessing.ipynb](./2_lisst_preprocessing.ipynb)
This notebook was run for each dataset and executes following steps:
* Reads in the in-situ LISST-200X and OBS data (available on [MDA](https://doi.org/10.14284/572))
* Correct the depth coordinate of the in-situ measurements to Lowest Astronomical Tide (LAT) using tide correction data from the vessel's underway data system and hydrographical data from [MDK](https://www.agentschapmdk.be/en/hydrographical-data).
* Visual inspection of the grain size distributions to:
    * remove of erroneous size classes and recalculate the Total Volume Concentration if required.
    * determine the boundaries for grain size populations
* Calculate grain size statistics using a python translation of the [G2Sd R package](https://cran.r-project.org/web/packages/G2Sd/), itself based on Gradistat (Blott and Pye, 2001).
* Export the pre-processed in-situ data to a new CSV file

## 3. Extract water column Sv data at in-situ sensor locations and model the relationship: [3_extract_and_model.ipynb](./3_extract_and_model.ipynb)
> disclaimer: This step is compute-intensive and was executed on the high-performance computing (HPC) environment of [VSC](https://www.vscentrum.be/). The preprocessed files generated in previous notebooks were copied onto the storage system of this HPC environment. 

This notebook was ran for each modelled parameter, with its set of suitable campaign data. TVC (TVC1-500µm, TVC1-3µm, TVC3-20µm, TVC20-200µm, TVC200-500µm) was only modeled based on data from campaigns 20-690, 21-430, 21-550_KW and 21-550_WD. The modeling of optical turbidity (in NTU) was based on data from campaigns 20-690, 21-092, 21-160. This notebook includes following steps.
* It configures the HPC cluster that will be used by the Dask scheduler
* It reads the pre-processed in-situ data and prepares it to be matched onto the water column data. This includes:
    * Re-project the coordinates of in-situ measurements to the same CRS as the MBES pointcloud
    * Ensure that the in-situ measurements fall within the octree index of the MBES pointcloud
    * Plot in-situ measurements and MBES pointcloud outlines for visual inspection
* Split the dataset in train and test data
* Build and Sklearn pipeline that:
    * Extracts and averages MBES Sv data in a sphere with a specific radius around the in-situ measurement
    * Imputes missing values 
    * Fits a linear regression model between the in-situ parameter and the Sv data.
* Run the sklearn pipeline using cross-validation to find the optimal radius of the spheres (hyperparameter)
* Print the parameters and plot the regression line for the model with optimal radius
* Asses the R2 of the linear regression model with optimal radius using the test dataset
* Save the linear regression model with optimal radius

## 4. Grid the water column Sv MBES data: [4_grid.ipynb](./4_grid.ipynb)
This notebook was run for each MBES dataset and executes following steps:
* It configures the Dask scheduler to run on the local machine
* It generates a 3D (XYZ) grid from the MBES pointcloud data in Entwine
* It saves the grid to a netCDF file
* It plots cross sections of the grid for visual inspection

## 5. Predict TVC from the gridded Sv MBES data, using linear regression models: [5_predict.ipynb](./5_predict.ipynb)
This notebook executes the following steps:
* For each grid file:
    * Load model files
    * For each linear regression model:
        * Predict the in-situ sensor parameter for each grid cell
    * Save the predicted grid to a netcdf file

## 6. Conversion of the volume to mass concentrations: [6_calculate_assemblage_apparent_density.ipynb](./6_calculate_assemblage_apparent_density.ipynb)
# 
This notebook executes the following steps:
* Calculate a theoretical apparent densities for the LISST-200X size classes using the formula from Fall et al. 2021. 
* Average the densities of the different size classes to match the grain size populations
* Convert the predicted TVC grids to Suspended Particulate Matter Concentration (SPMC) grids.
* Saves the SPMC grids to netCDF files

## 7. Publication outputs: [7_publication_outputs.ipynb](./7_publication_outputs.ipynb)
This notebook extracts plots and tables from the model and grid files, which were cosmetically improved prior to inclusion in the publication.
It includes:
* Regression model parameter tables and regression plots
* LISST-200X Particle Size distribution plot
* Optical misalignment plot
* Summary plot and table of the in-situ data
* Predicted SPMC map and vertical cutaway plot
* Comparison plot of predicted SPMC profiles with Niskin Bottle SPMC station measurements
* Colorbar for Potree viewer plot
* Optical Misalignment plot
* Table of predicted SPMC statistics per campaign

This research was conducted in the framework of the TIMBERS project (grant number SR/00/381), the STURMAPSS (dissemination) project (grant number SR/L9/221) and the TURBEAMS project (grant number RV/21/TURBEAMS). All projects were funded by the Belgian Science Policy Office BELSPO.