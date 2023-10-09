import statsmodels.api as sm
import pandas as pd
from timbers_code.entwine_extract import *

def fit_ols_model(df, x_dimension, y_dimension):
    '''
    takes in a dataframe df and returns an OLS statsmodel 
    y_dimension = beta_1 * x_dimension + beta_0 + error
    '''
    X = sm.add_constant(df[x_dimension])
    y = df[y_dimension]
    model = sm.OLS(y, X).fit()
    return model

def fit_models(df, radius, x_dimension, y_dimensions):
    '''
    Fit an ols model for each 
    '''
    models = {'radius':radius}
    for y_dimension in y_dimensions:
        models[y_dimension] = fit_ols_model(df, x_dimension, y_dimension)
    return models

def fit_models_for_radia(gdf, entwine_path, radia, x_dimension, x_dimension_statistic, y_dimensions, x_dimension_range = None):
    ''' 
    Takes in a point geodataframe df and a path to an entwine point tile directory
    For each radius in radia, it calculates an OLS model between x_dimension_statistic of x_dimension and each y_dimension in y_dimensions
    A matrix of models for each radius and y_dimension is returned
    '''
    model_matrix = []
    for radius in radia:
        print('running radius: '+str(radius))
        # get x_dimension average in sphere around each point
        pointcloud_stats = gather_stats_from_pointcloud_dask(entwine_path=entwine_path,
                                                        index = gdf.index,
                                                    points=gdf.geometry,
                                                    origin_files=gdf.linename,
                                                    distance=radius,
                                                    dimension=x_dimension)       
        df = pd.concat([gdf, pointcloud_stats.add_prefix('value_db_')],axis=1)
        # remove rows where the x_dimension statistic is 0
        df = df[df[x_dimension+'_'+x_dimension_statistic] != 0]
        # remove rows where the x_dimension statistic is NaN
        df = df[~df[x_dimension+'_'+x_dimension_statistic].isna()]
        # remove outliers in case a dimension range is provided
        if x_dimension_range is not None:
            df = df[df[x_dimension+'_'+x_dimension_statistic].between(left = x_dimension_range[0], right=x_dimension_range[1])]
        # add models to output
        model_matrix.append(fit_models(df, radius, x_dimension+'_'+x_dimension_statistic, y_dimensions))
    return pd.DataFrame(model_matrix).set_index('radius')