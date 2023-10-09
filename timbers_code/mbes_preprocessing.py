import dask.dataframe as dd
def preprocess_mbes_file(raw_file_path,
                         processed_file_path,
                         columns_map = {'Lon':'X','Lat':'Y','Depth':'Z','Value': 'value_db'},
                         sep = ',',
                         na_values = [],
                         crop_bbox = (),
                         color = True,
                         cmap_name = 'jet',
                         value_db_min = -100,
                         value_db_max = 0):
    '''
    This function takes in a MBES water column point data csv file exported form SonarScope, pre-processes it and writes to a csv file that can be ingested into entwine.
    Returns a dictionary with the number of rows in the processed file as well as the path to the processed file.
    Arguments:
    - raw_file_path: path from which to read original csv file with point data 
    - processed_file_path: path to write the processed csv file to
    - columns_map: dictionary that maps the header of the original file to entwine recognized position headers (X,Y,Z) as well as additional dimensions (e.g. value_db). Headers not included are removed.
    - sep: column separator of original file, passed to dask.dataframe.read_csv
    - na_values: a list of strings to parse as NaN's which will cause the containing rows to be removed, passed to dask.dataframe.read_csv. Use this in case there are issues with the csv file which return an error (misplaced column seperators etc.)
    - crop_bbox: If set, the points are cropped in plan view to the bounding box in the form of a tuple with (xmin,ymin,xmax,ymax)
    - color: If True, RGBA values are calculated for each point using the cmap_name color ramp scaled from value_db_min to value_db_max
    - cmap_name: the name of the color ramp to use
    - value_db_min: minimum value for the color ramp
    - value_db_max: maximum value for the color ramp
    '''
    # read selected columns from raw file path adding na_values in case of wrong symbols
    ddf = dd.read_csv(raw_file_path, sep = sep, usecols = list(columns_map.keys()), skipinitialspace = True, na_filter = True, skip_blank_lines=True, na_values = na_values)
    # # reorder columns and rename to match Entwine EPT headers
    ddf = ddf[list(columns_map.keys())]
    ddf = ddf.rename(columns = columns_map)
    # remove points with missing values
    ddf = ddf.dropna()
    # crop outlines to a specific marine region given by mrgrid
    if len(crop_bbox) == 4:
        ddf = ddf[ddf['X'].between(crop_bbox[0],crop_bbox[2]) & ddf['Y'].between(crop_bbox[1],crop_bbox[3])]
    # calculate the number of rows
    rows = ddf.shape[0]
    # add RGBA values to table if color is set
    if color:
        # create an RGBA mapper and apply to the data frame
        cmap = Dim2RgbaMapper(cmap_name, value_db_min, value_db_max)
        ddf = ddf.map_partitions(cmap.assign_rgba_to_df, dimension = 'value_db')
    # write  as single file to processed_file_path
    processed_file = ddf.to_csv(processed_file_path, index=False, single_file = True)[0]
    # return row names and output file as pandas series
    return {'rows':rows.compute(), 'processed_file':processed_file}

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
class Dim2RgbaMapper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
    def colorbar(self, dimension):
        fig,ax = plt.subplots(figsize = (5,1))
        fig.colorbar(self.scalarMap, cax=ax, orientation='horizontal', label=dimension)
    def assign_rgba_to_df(self, df, dimension):
        rgba = (self.scalarMap.to_rgba(df[dimension])*256).astype(int)
        return df.assign(Red=rgba[:,0],Green=rgba[:,1],Blue=rgba[:,2],Alpha= rgba[:,3])
