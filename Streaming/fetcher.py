import argparse
import datetime

from utils import connect_to_api, extract_district_polygon, convert_geojson_to_WKT

def fetch_products(api, args):
    product_type = "S2MSI{}".format(args.level)

    format_str = '%d%m%Y'
    start_date = args.start_date
    end_date = args.end_date
    start_date = datetime.datetime.strptime(start_date, format_str)
    end_date = datetime.datetime.strptime(end_date, format_str)

    requested_footprint = extract_district_polygon(args.district_file, args.district_name)
    requested_footprint = convert_geojson_to_WKT(requested_footprint['type'].values[0], requested_footprint['coordinates'].values[0])
    products = api.query(requested_footprint, date=(start_date, end_date), platformname='Sentinel-2', producttype=product_type)

    return products

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-file', type=str, default='SentinelAPIUser.json')
    parser.add_argument('--district-file', type=str, default='yemen_admin2_points.topojson')
    parser.add_argument('--district-name', type=str, required=True)
    parser.add_argument('--start-date', type=str, required=True, help="start date format dd/mm/YYYY")
    parser.add_argument('--end-date', type=str, required=True, help="end date format dd/mm/YYYY")
    parser.add_argument('--level', type=str, default='1C', help="1C or 2A")
    parser.add_argument('--path', type=str, default='data/unprocessed')
    args = parser.parse_args()

    api = connect_to_api(args.api_file)

    products = fetch_products(api, args)

    #api.download_all(products, directory_path=args.path)