#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:06:45 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""

import os
import pycpd
import numpy as np
import xmltodict
import matplotlib.pyplot as plt

from functools import partial

def read_annotations_xml(xml_dir:str) -> np.array:
    with open(xml_dir, 'r') as xml:
        xml_data = xml.read()
        
    xml_dict = xmltodict.parse(xml_data)
    out_points = []
    
    for points_dict in xml_dict['annotations']['image']['points']:
        out_points.extend(
            [[float(pnt) for pnt in _.split(',')] 
             for _ in points_dict['@points'].split(';')])
    
    return np.array(out_points)

def visualize_callback(iteration, error, X, Y, ax):
    plt.cla()
    print(iteration)
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def visualize_points(X, Y):
    fig = plt.figure(figsize=(18,9))
    
    plt.subplot(121)
    plt.scatter(X[:, 1],  X[:, 0], color='red', marker='o')
    plt.title('Target')
    
    plt.subplot(122)
    plt.scatter(Y[:, 1],  Y[:, 0], color='red', marker='o')
    plt.title('Source')
    
    plt.show()

if __name__ == '__main__':
    
    # ground trouth points
    gcp_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/GCP_11_4229.csv'
    ground_points_arr = np.genfromtxt(gcp_dir, delimiter=',', skip_header=1)
    ground_points_arr = ground_points_arr[:,:2]
    
    # points from picture annotation
    xml_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/punkty_11_4229/annotations.xml'
    img_points_arr = read_annotations_xml(xml_dir)
    
    visualize_points(ground_points_arr, img_points_arr)
    
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize_callback, ax=fig.axes[0])
    
    
    reg = pycpd.AffineRegistration(
        X=ground_points_arr, Y=img_points_arr)
    # run the registration & collect the results
    reg.register(callback)
    plt.show()
    
    visualize_points(ground_points_arr, TY)
    

    TY = reg.transform_point_cloud(TY)
    
    result = np.hstack((img_points_arr, TY))
    
    np.savetxt('/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/punkty_11_4229/match_table.txt',
               result, delimiter="     ", fmt='%.6f')
    
    
    
    
    
    
    
    
