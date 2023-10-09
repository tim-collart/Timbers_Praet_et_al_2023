import os
import re
def create_folder_if_absent(folder):
    '''
    creates folder if it doesn't exist
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)

        
import requests
from pyproj import Transformer        
def get_bbox_fom_marineregions(mrgid, srs = 'EPSG:4326'):
    '''
    Returns the bounding box (xmin, ymin, xmax, ymax) for a Marine Region with Marine Region ID mrgid in the Coordinate Reference System given by srs
    '''
    transformer = Transformer.from_crs('EPSG:4326',srs,always_xy=True) #crs of marine regions is WGS 84 = EPSG4326
    response = requests.get("https://www.marineregions.org/rest/getGazetteerRecordByMRGID.json/{}/".format(mrgid)).json()
    xmin,ymin=transformer.transform(response['minLongitude'], response['minLatitude'])
    xmax,ymax=transformer.transform(response['maxLongitude'], response['maxLatitude'])
    return (xmin,ymin,xmax,ymax)

def ddm2dec(dms_str):
    """Return decimal representation of DDM (degree decimal minutes)
    
    >>> ddm2dec("45° 17.896' N")
    48.8866111111F
    
    code credit to https://gis.stackexchange.com/questions/398021/converting-degrees-decimal-minutes-to-decimal-degrees-in-python
    """
    
    dms_str = re.sub(r'\s', '', dms_str)
    
    sign = -1 if re.search('[swSW]', dms_str) else 1
    
    numbers = [*filter(len, re.split('\D+', dms_str, maxsplit=4))]

    degree = numbers[0]
    minute_decimal = numbers[1] 
    decimal_val = numbers[2] if len(numbers) > 2 else '0' 
    minute_decimal += "." + decimal_val

    return sign * (int(degree) + float(minute_decimal) / 60)