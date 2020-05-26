from sentinelsat import SentinelAPI
import json
import pandas as pd

def connect_to_api(file) :
    with open(file, 'r') as f:
        user_dict = json.load(f)

    user = user_dict[0]['user']
    password = user_dict[0]['pass']
    api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')
    return api

def extract_district_polygon(jsonfile, district_name) :
    with open(jsonfile) as f:
        district_polygon = json.load(f)

    district_polygon = pd.DataFrame(data=district_polygon, columns=["district", "WKT"])
    district_geojson = district_polygon.loc[district_polygon['district'] == district_name]

    return district_geojson['WKT'].values[0]
