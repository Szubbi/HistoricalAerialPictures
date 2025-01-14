#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:30:45 2024

@author: piotr.szubert@doctoral.uj.edu.pl

"""

import Metashape

def import_markers_csv():
    doc = Metashape.app.document
    if not len(doc.chunks):
        print("No chunks. Script aborted.\n")
        return False

    path = Metashape.app.getOpenFileName("Select markers import file:")
    if path == "":
        print("Incorrect path. Script aborted.\n")
        return False

    print("Markers import started.\n")
    file = open(path, "rt")
    eof = False
    line = file.readline()  # skipping header line
    line = file.readline()
    if not len(line):
        eof = True

    chunk = doc.chunk
    while not eof:
        sp_line = line.strip().rsplit(",", 3)
        y = float(sp_line[2])
        x = float(sp_line[1])
        label = sp_line[0]

        flag = 0
        for camera in chunk.cameras:
            if camera.label.lower() == label.lower():
                marker = chunk.addMarker()
                marker.projections[camera] = Metashape.Marker.Projection(Metashape.Vector([x, y]), True)
                flag = 1
                break

        if not flag:
            print(f"Camera {label} not found.\n")

        line = file.readline()
        if not len(line):
            eof = True

    file.close()
    print("Markers import finished.\n")

import_markers_csv()
