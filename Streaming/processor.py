import os
import argparse
import rasterio as rio

from fetcher import fetch_products
from utils import connect_to_api, unzip, split_into_tiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-file', type=str, default='SentinelAPIUser.json')
    parser.add_argument('--district-file', type=str, default='yemen_admin2_points.topojson')
    parser.add_argument('--district-name', type=str, required=True)
    parser.add_argument('--start-date', type=str, required=True, help="start date format dd/mm/YYYY")
    parser.add_argument('--end-date', type=str, required=True, help="end date format dd/mm/YYYY")
    parser.add_argument('--level', type=str, default='1C', help="1C or 2A")
    parser.add_argument('--path', type=str, default='data/unprocessed')
    parser.add_argument('--create-database', action='store_true') #TODO CREATE DATABASE
    args = parser.parse_args()

    api = connect_to_api(args.api_file)

    products = fetch_products(api, args)
    products_df = api.to_dataframe(products)

    best_product = products_df[products_df.cloudcoverpercentage == products_df.cloudcoverpercentage.min()]

    api.download(best_product.iloc[0]['uuid'], directory_path=args.path)

    unzip(args.path, best_product.iloc[0]['identifier'])

    imagePath = '{}/{}.SAFE/GRANULE/'.format(args.path, best_product.iloc[0]['identifier'])
    imagePath = '{}{}/IMG_DATA/'.format(imagePath, os.listdir(imagePath)[0])  # using index 1 due to DS.Store
    imagePath = '{}{}'.format(imagePath, "_".join(os.listdir(imagePath)[0].split("_", 2)[:2]))

    band2 = rio.open(imagePath + '_B02.jp2', driver='JP2OpenJPEG')  # blue
    band3 = rio.open(imagePath + '_B03.jp2', driver='JP2OpenJPEG')  # green
    band4 = rio.open(imagePath + '_B04.jp2', driver='JP2OpenJPEG')  # red
    band8 = rio.open(imagePath + '_B08.jp2', driver='JP2OpenJPEG')  # nir

    # export true color image
    trueColor = rio.open('data/TC/{}_tc.tiff'.format(best_product.iloc[0]['identifier']), 'w', driver='Gtiff',
                              width=band4.width, height=band4.height,
                              count=3,
                              crs=band4.crs,
                              transform=band4.transform,
                              dtype=band4.dtypes[0]
                              )
    trueColor.write(band2.read(1), 3)  # blue
    trueColor.write(band3.read(1), 2)  # green
    trueColor.write(band4.read(1), 1)  # red
    trueColor.close()

    # export false color image
    falseColor = rio.open('data/FC/{}_fc.tiff'.format(best_product.iloc[0]['identifier']), 'w', driver='Gtiff',
                               width=band2.width, height=band2.height,
                               count=3,
                               crs=band2.crs,
                               transform=band2.transform,
                               dtype='uint16'
                               )
    falseColor.write(band3.read(1), 3)  # Blue
    falseColor.write(band4.read(1), 2)  # Green
    falseColor.write(band8.read(1), 1)  # Red
    falseColor.close()

    split_into_tiles('data/TC/{}_tc.tiff'.format(best_product.iloc[0]['identifier']), 'data/processed/',
                     best_product.iloc[0]['identifier']+"_tc", 4)

    '''split_into_tiles('data/FC/{}_fc.tiff'.format(best_product.iloc[0]['identifier']), 'data/processed/',
                     best_product.iloc[0]['identifier'] + "_fc", 4)'''