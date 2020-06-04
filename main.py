import argparse

import Augmentation
from Augmentation import augment
import Streaming
from Streaming import fetcher, processor

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-file', type=str, default='Streaming/SentinelAPIUser.json')
    parser.add_argument('--district-file', type=str, default='Streaming/yemen_admin2_points.topojson')
    parser.add_argument('--district-name', type=str, required=True)
    parser.add_argument('--start-date', type=str, required=True, help="start date format dd/mm/YYYY")
    parser.add_argument('--end-date', type=str, required=True, help="end date format dd/mm/YYYY")
    parser.add_argument('--level', type=str, default='1C', help="1C or 2A")
    parser.add_argument('--unprocessed-path', type=str, default='Streaming/data/unprocessed')
    parser.add_argument('--processed-path', type=str, default='Augmentation/data/processed')
    parser.add_argument('--hdf5-file', type=str, default='Augmentation/processed_dataset.h5')
    parser.add_argument('--augmented-path', type=str, default='Detection/data/images')
    parser.add_argument('--model', type=str, default="SRCNN", help='SRCNN, Subpixel, FSRCNN')
    parser.add_argument('--augmentation-weights-file', type=str, default="Augmentation/outputs/x4/best.pth")
    parser.add_argument('--create-database', action='store_true')# call create database function from processor
    args = parser.parse_args()

    # Fetch the data from sentinel-2 using api
    api = Streaming.utils.connect_to_api(args.api_file)
    products = fetcher.fetch_products(api, args)

    # Pre-processing the data before being augmented
    products_df = api.to_dataframe(products)
    best_product = products_df[products_df.cloudcoverpercentage == products_df.cloudcoverpercentage.min()]
    api.download(best_product.iloc[0]['uuid'], directory_path=args.unprocessed_path)
    processor.process_products(args.unprocessed_path, args.processed_path, best_product)

    # TODO Augment
    augment.prepare(args)
    # Create the hdf5 dataset
    dataset = []
    augment.augment(args, dataset)
    # --------
    # TODO Add Detection module