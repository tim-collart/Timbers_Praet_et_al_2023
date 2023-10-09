import pandas as pd

def gapstime2datetime(date, gaps_time):
    '''
    takes in a date string of "YYYY-MM-DD" format
    and a time string formatted as HHMMSS.SSS float as provided by GAPS
    and returns datetime object
    '''
    hours = (gaps_time // 10000).astype(int).astype(str).str.zfill(2)
    minutes = ((gaps_time % 10000) // 100).astype(int).astype(str).str.zfill(2)
    seconds = (gaps_time % 100).round(3).astype(str)
    datestring = date+'T'+hours+':'+minutes+':'+seconds
    return pd.to_datetime(datestring)

def gapscoordinate2dd(gaps_coordinate):
    '''
    takes in coordinate formatted as DDMM.MMM float (degrees and decimal minutes) as provided by GAPS
    and returns decimal degrees
    '''
    degrees = (gaps_coordinate // 100)
    decimalminutes = (gaps_coordinate % 100)
    decimaldegrees = degrees + decimalminutes/60
    return decimaldegrees

def formatgaps(gaps_df, date):
    '''
    takes in a gaps dataframe with a 'time','longitude','latitude' and 'depth' column and a date string of "YYYY-MM-DD" format
    formats date and gaps time to datetime, latitude and longitude to decimal minutes
    resamples the timeseries to 1 second and interpolates values if data is missing for that second
    '''
    gaps_df.time = gapstime2datetime(date, gaps_df.time)
    gaps_df = gaps_df.set_index('time')
    gaps_df.longitude = gapscoordinate2dd(gaps_df.longitude)
    gaps_df.latitude = gapscoordinate2dd(gaps_df.latitude)
    return gaps_df.resample('1s').mean().interpolate(method='time')

def add_linename_on_timestamp(df,lines):
    '''
    takes in a dataframe df with a timeindex and a df lines which has linename, start time and stop time
    and adds a linename column to df based on the time index
    '''
    for idx, linename, start, stop in lines.itertuples():
        df.loc[(df.index >= start) & (df.index <= stop),'linename'] = linename
    return df