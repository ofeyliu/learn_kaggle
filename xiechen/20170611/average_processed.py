#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import sys
import string

file_params = []

for i in range( len(sys.argv) - 1 ):
    file_params.append( sys.argv[i+1] )

csv_result = file( 'result_processed.csv', 'w' )
write_upload = csv.writer( csv_result )


file_num = len( file_params )

arrages = []
for i in range( file_num ):
    
    file_open = file( file_params[i], 'r' )
    reader = csv.reader( file_open )
    if i == 0:
        arrage1 = []
        arrage2 = []
        arrage3 = []
        for index, line in enumerate( reader ):
            arrage1.append(line[0])
            arrage2.append(line[1])
            arrage3.append(line[2])

        arrages.append(arrage1)
        arrages.append(arrage2)
        arrages.append(arrage3)
    else:
        arrage = []
        for index, line in enumerate( reader ):
            arrage.append( line[2] )
        arrages.append(arrage)

    file_open.close()

print(len( arrages[1] ))

average_arrage = []

for j in range( len(arrages[0]) ):
    if j == 0 :
        write_upload.writerows([( arrages[0][j], arrages[1][j], 'Average_Probabilites' )])
    else:
        average = ( string.atof( arrages[2][j] ) + 
                    string.atof( arrages[3][j] ) + 
                    string.atof( arrages[4][j] ) + 
                    string.atof( arrages[5][j] ) +
                    string.atof( arrages[6][j] ) ) / 5.0

        write_upload.writerows([( arrages[0][j], arrages[1][j], average )])

csv_result.close()







