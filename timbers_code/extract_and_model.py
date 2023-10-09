import pdal
import os
from subprocess import Popen, PIPE
import tempfile
import ujson as json
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely.wkt
import dask
import dask.dataframe as dd
import xarray as xr
import uuid
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from joblib import load
import statsmodels.api as sm


def get_outline_from_pointcloud(entwine_path, origin_file = None, edge_size = 0.5):
    '''
    PDAL pipeline that reads points from entwine_path,
    calculates the outline of the pointcloud and returns it as a shapely object
    '''
    pipe= [
        {
            "type":"readers.ept",
            "filename":entwine_path+'/ept.json'
        },
        {
            "type" : "filters.hexbin",
            "edge_size" : edge_size
        }
    ]
    if origin_file is not None:
        pipe[0]["origin"] = origin_file
    pipeline = pdal.Pipeline(json.dumps(pipe))
    pipeline.loglevel = 8 #really noisy
    pipeline.validate()
    pipeline.execute()
    return shapely.wkt.loads(json.loads(pipeline.metadata)['metadata']['filters.hexbin'][0]['boundary'])

def get_outline_from_pointcloud_stream(entwine_path, origin_file = None, edge_size = 0.5):
    '''
    PDAL info command to extract the outline of the point cloud
    The resulting polygon, with edge length edge_size is returned as a shapelyobject
    '''
    pipe= [
        {
            "type":"readers.ept",
            "filename":entwine_path+'/ept.json'
        },
        {
            "type" : "filters.hexbin",
            "edge_size" : edge_size 
        }
    ]
    if origin_file is not None:
        pipe[0]["origin"] = origin_file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        p = Popen(['pdal','pipeline','--stdin','--stream','--metadata',f.name], stdin=PIPE)
        p.communicate(input=json.dumps(pipe).encode())
        metadata = json.loads(f.read())
        metadatafile = f.name
    os.remove(metadatafile)
    return shapely.wkt.loads(metadata['stages']['filters.hexbin']['boundary'])

   
def get_base_extract_pipe(entwine_path, point, distance, origin_file = None):
    '''
    PDAL pipeline that reads points from entwine,
    crops them to sphere with center point and radius distance
    If supplied, the points are filtered to those extracted from the origin_file
    '''
    pipe=[
        {
            "type":"readers.ept",
            "filename":entwine_path+'/ept.json',
            "bounds":"("+ \
                     "[{},{}]".format(point.x-distance, point.x+distance) + \
                     "," + \
                     "[{},{}]".format(point.y-distance, point.y+distance) + \
                     "," + \
                     "[{},{}]".format(point.z-distance, point.z+distance) + \
                     ")"
        },
        {
            "type":"filters.crop",
            "point": point.wkt,
            "distance": distance
        }
    ]
    if origin_file is not None:
        pipe[0]["origin"] = origin_file
    return pipe
    
## The code below uses the python api to pdal but this does not support all ept reader options
# def pdal_extract(entwine_path, point, distance, dimension, origin_file = None):
#     '''
#     PDAL pipeline that reads points from entwine,
#     crops them to sphere with center point and radius distance,
#     and calculates statistics of dimension
#     '''
#     pipe= [
#         {
#             "type":"readers.ept",
#             "filename":entwine_path+'/ept.json',
#             "bounds":"("+ \
#                      "[{},{}]".format(point.x-distance, point.x+distance) + \
#                      "," + \
#                      "[{},{}]".format(point.y-distance, point.y+distance) + \
#                      "," + \
#                      "[{},{}]".format(point.z-distance, point.z+distance) + \
#                      ")"
#         },
#         {
#             "type":"filters.crop",
#             "point": point.wkt,
#             "distance": distance
#         },
#         {
#             "type":"filters.stats",
#             "dimensions":dimension
#         }
#     ]
#     if origin_file is not None:
#         pipe[0]["origin"] = origin_file
#     print(pipe)
#     pipeline = pdal.Pipeline(json.dumps(pipe))
#     pipeline.loglevel = 8 #really noisy
#     pipeline.validate()
#     pipeline.execute()
#     return pipeline

def extract_stats_in_sphere(entwine_path, point, distance, dimension, origin_file=None):
    '''
    Extends the base PDAL extract pipeline with a filter that calculates the statics of the dimension in the sphere
    Executes the pipeline and returns the statistics as a dictionnary
    '''
    # get base pipe
    pipe = get_base_extract_pipe(entwine_path, point, distance, origin_file)
    # open a temporary file to write to
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        # add the filter that calculates the stats
        pipe.append(
            {
                "type":"filters.stats",
                "dimensions":dimension
            }        
        )
        pipe.append(
            {
                "type":"writers.null"
            }        
        )
        p = Popen(['pdal','pipeline','--stdin','--metadata',f.name], stdin=PIPE)
        p.communicate(input=json.dumps(pipe).encode())
        metadatafile = f.name
        try:
            metadata = json.loads(f.read())['stages']['filters.stats']['statistic'][0]
        except Exception as e:
            warnings.warn('No stats could be read for point:{} as a result of the following exception {}. Passing NaNs '.format(point.wkt, e))
            metadata = {}
    os.remove(metadatafile)
    return metadata


def gather_stats_from_pointcloud_dask(distance, dimension, index, points, entwine_paths, origin_files=None):
    '''
    This function runs the extract_stats_in_sphere for each point in points in parrallel using Dask.
    Points should be a geoseries object with an index and a geometry
    Returns a pandas dataframe with the statistics.
    '''
    lazy_results = []
    for point, entwine_path, origin_file in zip(points, entwine_paths, origin_files):
        lazy_results.append(dask.delayed(extract_stats_in_sphere)(entwine_path=entwine_path,
                                                                  point=point,
                                                                  distance=distance,
                                                                  dimension=dimension,
                                                                  origin_file=origin_file))
    results = dask.compute(*lazy_results)
    results = pd.DataFrame(results).set_index(index)
    return results

def gather_stats_from_pointcloud_loop(distance, dimension, index, points, entwine_paths, origin_files=None):
    '''
    This function runs the extract_stats_in_sphere for each point in points in a loop.
    Points should be a geoseries object with an index and a geometry
    Returns a pandas dataframe with the statistics.
    '''
    results = pd.DataFrame()
    for point, entwine_path, origin_file in zip(points, entwine_paths, origin_files):
        results = results.append(extract_stats_in_sphere(entwine_path=entwine_path,
                                                              point=point,
                                                              distance=distance,
                                                              dimension=dimension,
                                                              origin_file=origin_file),
                                 ignore_index=True)
    results = results.set_index(index)
    return results


@dask.delayed
def extract_points_in_sphere(entwine_path, index, point, distance, dtype, origin_file=None):
    '''
    Extends the base PDAL extract pipeline with a writer that outputs all the extracted points to a csv file
    Executes the pipeline, reads the data from the csv files, and returns it as a pandas DataFrame with as index the initial point provided.
    '''
    # get base pipe
    pipe = get_base_extract_pipe(entwine_path, point, distance, origin_file)
    # open a temporary file to write to
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        # add the writer that writes to the temporary file
        pipe.append(
            {
                "type":"writers.text",
                "format":"csv",
                "order":",".join(dtype.keys()),
                "keep_unspecified":"false",
                "filename": f.name
            }        
        )
        p = Popen(['pdal','pipeline','--stdin'], stdin=PIPE)
        p.communicate(input=json.dumps(pipe).encode())
        datafile = f.name
        df = pd.read_csv(datafile, dtype = dtype)
        df.index = [index]*len(df)
    os.remove(datafile)
    return df


def gather_points_from_pointcloud(entwine_paths, index, points, distance, origin_files=None, dtype = {'Alpha':  np.int32,
                                                                                                     'Blue':  np.int32,
                                                                                                     'Green':  np.int32,
                                                                                                     'OriginId':  np.int32,
                                                                                                     'Red':  np.int32,
                                                                                                     'X': np.float64,
                                                                                                     'Y': np.float64,
                                                                                                     'Z': np.float64,
                                                                                                     'value_db': np.float64}):
    '''
    This function runs the extract_points_in_sphere for each point in points.
    Points should be a geoseries object with an index and a geometry
    Returns a pandas dataframe with all the points
    '''
    lazy_results = []
    # consider adding index so that this can be passed to dask
    for i, point, entwine_path, origin_file in zip(index, points, entwine_paths, origin_files):
        lazy_results.append(extract_points_in_sphere(entwine_path=entwine_path,
                                                     index = i,
                                                     point=point,
                                                     distance=distance,
                                                     dtype = dtype,
                                                     origin_file=origin_file))
    # we return it as a dask dataframe on which we can later do operations
    results = dd.from_delayed(lazy_results, meta = dtype)
    return results

class EntwineStatsExtractorLoop(BaseEstimator, TransformerMixin):
    # List of features in 'feature_names' and the 'power' of the exponent transformation
    def __init__(self, distance, dimension, statistic):
        self.distance = distance
        self.dimension = dimension
        self.statistic = statistic
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        output = gather_stats_from_pointcloud_loop(distance=self.distance,
                                                   dimension=self.dimension,
                                                   index = X.index,
                                                   points= X.geometry,
                                                   entwine_paths = x.entwine_path,
                                                   origin_files=X.linename)
        # if no points were in the radius, replace 0's with nan's, this will throw an error in the estimator
        # print('No points for ' + str((output['count'] == 0).sum()) + ' sample(s)')
        output[output['count'] == 0] = output[output['count'] == 0].replace(0,np.nan)
        # print(output[self.statistic].shape)
        # return the desired statistic
        return output[[self.statistic]]
    def inverse_transform(self, X):
        return X
    
class EntwineStatsExtractorParrallel(BaseEstimator, TransformerMixin):
    # List of features in 'feature_names' and the 'power' of the exponent transformation
    def __init__(self, distance, dimension, statistic):
        self.distance = distance
        self.dimension = dimension
        self.statistic = statistic
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        output = gather_stats_from_pointcloud_dask(distance=self.distance,
                                                   dimension=self.dimension,
                                                   index = X.index,
                                                   points= X.geometry,
                                                   entwine_paths=X.entwine_path,
                                                   origin_files=X.linename)
        # if no points were in the radius, replace 0's with nan's, this will throw an error in the estimator
        # print('No points for ' + str((output['count'] == 0).sum()) + ' sample(s)')
        output[output['count'] == 0] = output[output['count'] == 0].replace(0,np.nan)
        # print(output[self.statistic].shape)
        # return the desired statistic
        return output[[self.statistic]]
    def inverse_transform(self, X):
        return X

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self
    def predict(self, X, with_uncertainty = False):
        if self.fit_intercept:
            X = sm.add_constant(X)
        if with_uncertainty:
            return self.results_.get_prediction(X).summary_frame()
        return self.results_.predict(X)

@dask.delayed
def raster_from_entwine(entwine_path,
                      minx, maxx,
                      miny, maxy,
                      minz, maxz,
                      dimension = 'value_db',
                      statistic = 'mean',
                      radius = 10,
                      raster_resolution = 10,
                      origin_file = None,
                      tmp_raster_path = 'data/tmp'):
    '''
    PDAL pipeline that reads points from entwine_path within the bounds (minx, maxx, miny, maxy, minz, maxz) and creates a 2D raster  defined by minx, maxx, miny, maxy and the raster resolution, where each raster value is calculated by taking the statistic from the dimenions, in a cylinder with radius around the raster cell center. 
    Returns the file path to the raster in GeoTIFF (.tif) format stored in the tmp_raster_path
    '''
    # create a unique filepath to store the raster file in
    file_name = tmp_raster_path+'/'+str(uuid.uuid4())+'.tif'
    # generate a pipeline
    pipe=[
        {
            "type":"readers.ept",
            "filename":entwine_path+'/ept.json',
            "bounds":"("+ \
                 "[{},{}]".format(minx, maxx) + \
                 "," + \
                 "[{},{}]".format(miny, maxy) + \
                 "," + \
                 "[{},{}]".format(minz, maxz) + \
                 ")"

        }, 
        # {
        #     "type":"filters.range",
        #     "limits":"Z[{}:{}]".format(z_min,z_max)
        # },
        {
            "type": "writers.gdal",
            "bounds":"("+ \
                 "[{},{}]".format(minx, maxx) + \
                 "," + \
                 "[{},{}]".format(miny, maxy) + \
                 ")",
            "dimension": dimension,
            "power": 0.0, #turn off inverse distance weighting
            "output_type": statistic,
            "resolution": raster_resolution, # raster resolution
            "radius": radius, #radius around raster cell center from which points are averaged
            "gdaldriver": "GTiff",
            "filename":file_name
        }
    ]
    # add origin file if supplied
    if origin_file is not None:
        pipe[0]["origin"] = origin_file
    # run pdal pipeline in stream mode
    p = Popen(['pdal','pipeline','--stdin','--stream'], stdin=PIPE)
    p.communicate(input=json.dumps(pipe).encode())
    # read in the produced data file, squeeze the band coordinate, add the z coordinate and rename the band_data
    raster = xr.load_dataset(file_name, engine = 'rasterio').squeeze('band', drop = True).assign_coords({'z':(minz+maxz)/2.}).rename(name_dict = {'band_data':statistic+'_'+dimension})
    return raster


def get_raster_bounds(entwine_path):
    # Get Z bounds from entwine index
    with open(entwine_path+'/ept.json') as f:
        bounds = json.load(f)['boundsConforming']
    minz, maxz = bounds[2],bounds[5]
    # get x, y bounds from outline file
    # caclulate outline of of the pointcloud if it doesn't already exist in the entwine path
    if not os.path.exists(entwine_path+'/outline.json'):
        outline = get_outline_from_pointcloud_stream(entwine_path)
        gpd.GeoSeries([outline]).to_file(entwine_path+"/outline.json", driver="GeoJSON")
    else:
        outline = gpd.read_file(entwine_path+'/outline.json').geometry[0]
    minx, miny, maxx, maxy = outline.bounds
    return (minx, maxx, miny, maxy, minz, maxz)



def generate_raster_stack(entwine_path,
                          minx, maxx,
                          miny, maxy,
                          minz, maxz,
                          z_levels,
                          dimension = 'value_db',
                          statistic = 'mean',
                          radius = 10,
                          raster_resolution = 10,
                          origin_file = None,
                          tmp_raster_path = 'data/tmp'):
        '''
        Wraps raster_from_entwine and creates 2D rasters in parllel for z_levels
        Returns a stacks the reaster as a multidemensional xarray
        '''
        # create 2d rasters for each z level in parallel
        lazy_results = []
        for z_level in z_levels:
            lazy_results.append(raster_from_entwine(entwine_path = entwine_path,
                                                  minx = minx, maxx = maxx,
                                                  miny = miny, maxy = maxy,
                                                  minz = z_level.left, maxz = z_level.right,
                                                  dimension = dimension,
                                                  statistic = statistic,
                                                  radius = radius,
                                                  raster_resolution = raster_resolution,
                                                  origin_file = origin_file,
                                                  tmp_raster_path = tmp_raster_path))
        
        # compute the rasters
        rasters = dask.compute(*lazy_results)
        # return the rasters stacked along the z dimension
        return xr.concat(rasters, dim = 'z')
    
def predict_raster_from_model(raster_file,
                              model_file,
                              x_dimension,
                              output_raster_path,
                              exp_y_dimension = True,
                              stack_coords = ("x","y","z")):
    '''
    Takes in:
    - a raster_file pointing to a netcdf file with x_dimension along stack_coords
    - a model_file pointing to a Statsmodel model object
    Produces
    - a netcdf file with the predictions added to the original raster in output_raster_path
    '''
    # read in raster and stack the coords
    raster = xr.open_dataset(raster_file).stack(mi=stack_coords)
    # read in model an extract the y_dimension
    model = load(model_file)
    y_dimension = model._results.model.endog_names.replace(' ','_')
    # predic the raster values with the model
    predictions = model.get_prediction(sm.add_constant(raster[x_dimension].data)).summary_frame().add_prefix(y_dimension+'_')
    # add predictions into the raster
    for column in predictions.columns:
        if exp_y_dimension:
            raster[column.replace('Log10_','')] = ('mi', 10**predictions[column].to_numpy())
            y_dimension = y_dimension.replace('Log10_','')
        else:
             raster[column] = ('mi',predictions[column].to_numpy())
    output_file = f'{output_raster_path}/{os.path.splitext(os.path.basename(raster_file))[0]}_{y_dimension}.nc'
    raster.unstack('mi').to_netcdf(output_file)
    return output_file

# @dask.delayed
# def predict_raster_from_model_files(raster_file,
#                                     model_files,
#                                     output_raster_path,
#                                     x_dimension,
#                                     exp_y_dimension = True,
#                                     stack_coords = ("x","y","z")):
#     '''
#     Wraps predict_raster_from_model and runs it for each model in model_files
#     '''
#     raster = xr.open_dataset(raster_file)
#     y_dimensions = []
#     for model_file in model_files:
#         model = load(model_file)
#         # get the y dimension from the model
#         y_dimension = model._results.model.endog_names.replace(' ','_')
#         # created a predicted raster stack for each of the models
#         raster = predict_raster_from_model(raster=raster,
#                                            model=model,
#                                            x_dimension= x_dimension,
#                                            y_dimension= y_dimension,
#                                            exp_y_dimension = exp_y_dimension,
#                                            stack_coords = stack_coords)
#         y_dimensions.append(y_dimension)
#     # write the raster to a netCDF file
#     output_file = output_raster_path + os.path.splitext(os.path.basename(raster_file))[0]+'_'+'_'.join(y_dimensions)+'.nc'
#     raster.to_netcdf(output_file)
#     return(output_file)