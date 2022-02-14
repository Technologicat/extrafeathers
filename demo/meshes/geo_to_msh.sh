#!/bin/bash
for FILE in *.geo
do
    gmsh -2 "$FILE"
done
