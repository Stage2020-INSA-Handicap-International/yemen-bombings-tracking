# Streaming

Fetch and process the data used in our model

## Fetcher 
Fetcher is used to get the data from the satellite Sentinel-2.

- --api-file : used to initialize the connection with the sentinelsat api with a json file containing "user" and "pass". By default uses a document called SentinelAPIUser.json.
- --district-file : json file containing a list of all the districts (in our case all districts in Yemen) and their corresponding polygon
- --district-name : required in order to fetch the information of a particular district
- --start-date : self-explanatory. format ddmmYYYY
- --end-date : self-explanatory. format ddmmYYYY
- --level : Used to indicate what product type we are searching for. By default level is 1C (product type S2MSI1C) but level can be 2A (S2MSI2A)

example :
```sh
 python fetcher.py --district-name "Marib" --start-date "10052020" --end-date "16052020" --level 2A
```

## Utils
