#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 20:43:51 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""
import os
import requests
import geopandas as gpd
import pandas as pd
import json
import xml.etree.ElementTree as ET

from io import BytesIO
from shapely.geometry import Point, Polygon, LineString
from shapely.affinity import rotate

def download_wfs_polygons(url, layer_name):
    params = {
        'service': 'WFS',
        'version': '1.1.0',
        'request': 'GetFeature',
        'typename': layer_name,
        'outputFormat': 'text/xml; subtype=gml/3.1.1'
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()  # Check for request errors
    gdf = gpd.read_file(BytesIO(response.content))
    gdf.set_crs(crs = 'EPSG:2180', allow_override=True, inplace=True)
    
    return gdf

def get_wfs_layers(url):
    params = {
        'service': 'WFS',
        'version': '1.1.0',
        'request': 'GetCapabilities'
    }
    response = requests.get(url, params=params)
    response.raise_for_status()  # Check for request errors

    # Parse the XML response
    tree = ET.fromstring(response.content)
    layers = tree.findall('.//{http://www.opengis.net/wfs}FeatureType/{http://www.opengis.net/wfs}Name')
    layer_names = [layer.text for layer in layers]
    return layer_names

# reverse geom axis in gpd
def flip_geom(geom):
        if geom.geom_type == 'Point':
            flipped = Point(geom.y, geom.x)
            return flipped
        elif geom.geom_type == 'Polygon':
            exterior_coords = [(y, x) for x, y in geom.exterior.coords]
            interiors_coords = [[(y, x) for x, y in interior.coords] for interior in geom.interiors]
            flipped = Polygon(exterior_coords, interiors_coords)
            return flipped
        elif geom.geom_type == 'LineString':
            flipped = LineString([(y, x) for x, y in geom.coords])
            return flipped
        else:
            return geom

def download_img(src_url, dst_dir):
    image_name = src_url.spli('/')[-1]
    img_dst_dir = os.path.join(dst_dir, image_name)
    try:
        response = requests.get(src_url)
        response.raise_for_status()  # Check if the request was successful
        with open(img_dst_dir, 'wb') as file:
            file.write(response.content)
        print(f"Image successfully downloaded to {img_dst_dir}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")


if __name__ == "__main__":
    url = 'https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WFS/Skorowidze'
    layer_name = 'gugik:SkorowidzOrtofomapy2024'
    db_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/05_Data/Data.gpkg'
    training_areas_gdf = gpd.read_file(db_dir, layer = 'obszaryTestowe_01')
    columns = ['gml_id', 'godlo', 'akt_rok', 'piksel', 'kolor', 'zrodlo_danych',
               'uklad_xy', 'modul_archiwizacji', 'nr_zglosz', 'akt_data', 
               'czy_ark_wypelniony', 'url_do_pobrania', 'dt_pzgik', 'wlk_pliku_mb',
               'geometry']
    
    orto_gdf = gpd.GeoDataFrame(columns = columns, crs = "EPSG:2180")
    wfs_layers = get_wfs_layers(url)[::-1]
    not_overlapped_areas = training_areas_gdf
    
    # creating gdf with tiles to download
    for orto_layer in wfs_layers[:5]:
        orto_layer_gdf = download_wfs_polygons(url, orto_layer)
        orto_layer_gdf['geometry'] = orto_layer_gdf['geometry'].apply(flip_geom)
        print(f'pobrano: {orto_layer}')
        
        overlapping_tiles = orto_layer_gdf[
            orto_layer_gdf.intersects(not_overlapped_areas.union_all())]
        not_overlapped_areas = not_overlapped_areas[
            ~not_overlapped_areas.intersects(orto_layer_gdf.union_all())]
        
        orto_gdf = pd.concat([orto_gdf, overlapping_tiles], ignore_index=True)
        
        print(f'{len(overlapping_tiles)} added to Orto Tiles')
        print(f'{len(not_overlapped_areas)} more to go')
        
        
        if len(not_overlapped_areas) == 0:
            print('All test areas covered by Ortos')
            break
        
    orto_gdf.to_file(db_dir, layer = 'ortoDoPobrania_04')
    
    
    # downloading ortos 
    

    
