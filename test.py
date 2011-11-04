#!/usr/bin/python
#
# Copyright (c) 2011 Ginkgo Bioworks Inc.
# Copyright (c) 2011 Daniel Taub
#
# This file is part of Scantelope.
#
# Scantelope is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Test module for Scantelope.

Change MaxReps to dictate how many complete scans to do. 
"""

import scan
from datetime import datetime
from time import sleep
import threading

scanEvent = threading.Event()

MaxReps = 10

sc= scan.ScanControl(scanEvent)

#supports low res (300) or high (600) (in dpi)
sc.setNextRes(600)
sc.setResFromNext()

# if true, clears decoded before repeating
# if false, attempts to exhaustively decode the wells
sc.forceRepeat = True

# do not actually scan, but use images from most recent
#decodeOnly = True

sc.enableScan()
sc.start()

# each time set, allows a waiting scan to proceed
scanEvent.set() 

# If false, compare each decoding to 'comp' below
# updateFromImages = False
# If true, this allows decoding to the 'correct' decoding to be updated,
# testing for consistency across scans instead of a fixed correctness
updateFromImages = True

lastt = datetime.now()
out = {}
f=open('dmLog.txt','a')

# replace entry for 'barcode' with your barcodes to test
comp = '''TUBE,BARCODE,STATUS
A01,1111111111,OK
A02,1111111111,OK
A03,1111111111,OK
A04,1111111111,OK
A05,1111111111,OK
A06,1111111111,OK
A07,1111111111,OK
A08,1111111111,OK
A09,1111111111,OK
A10,1111111111,OK
A11,1111111111,OK
A12,1111111111,OK
B01,1111111111,OK
B02,1111111111,OK
B03,1111111111,OK
B04,1111111111,OK
B05,1111111111,OK
B06,1111111111,OK
B07,1111111111,OK
B08,1111111111,OK
B09,1111111111,OK
B10,1111111111,OK
B11,1111111111,OK
B12,1111111111,OK
C01,1111111111,OK
C02,1111111111,OK
C03,1111111111,OK
C04,1111111111,OK
C05,1111111111,OK
C06,1111111111,OK
C07,1111111111,OK
C08,1111111111,OK
C09,1111111111,OK
C10,1111111111,OK
C11,1111111111,OK
C12,1111111111,OK
D01,1111111111,OK
D02,1111111111,OK
D03,1111111111,OK
D04,1111111111,OK
D05,1111111111,OK
D06,1111111111,OK
D07,1111111111,OK
D08,1111111111,OK
D09,1111111111,OK
D10,1111111111,OK
D11,1111111111,OK
D12,1111111111,OK
E01,1111111111,OK
E02,1111111111,OK
E03,1111111111,OK
E04,1111111111,OK
E05,1111111111,OK
E06,1111111111,OK
E07,1111111111,OK
E08,1111111111,OK
E09,1111111111,OK
E10,1111111111,OK
E11,1111111111,OK
E12,1111111111,OK
F01,1111111111,OK
F02,1111111111,OK
F03,1111111111,OK
F04,1111111111,OK
F05,1111111111,OK
F06,1111111111,OK
F07,1111111111,OK
F08,1111111111,OK
F09,1111111111,OK
F10,1111111111,OK
F11,1111111111,OK
F12,1111111111,OK
G01,1111111111,OK
G02,1111111111,OK
G03,1111111111,OK
G04,1111111111,OK
G05,1111111111,OK
G06,1111111111,OK
G07,1111111111,OK
G08,1111111111,OK
G09,1111111111,OK
G10,1111111111,OK
G11,1111111111,OK
G12,1111111111,OK
H01,1111111111,OK
H02,1111111111,OK
H03,1111111111,OK
H04,1111111111,OK
H05,1111111111,OK
H06,1111111111,OK
H07,1111111111,OK
H08,1111111111,OK
H09,1111111111,OK
H10,1111111111,OK
H11,1111111111,OK
H12,1111111112,OK
'''

if not updateFromImages:
    for line in comp.strip().split('\n'):
        a,b,c = line.split(',')
        out[a] = b

notDone = True

while notDone:
    sleep(2)

    d = sc.getNewDecoded(lastt)

    if d == -1:
        continue
    else:
        MaxReps -= 1
        if MaxReps < 0:
            notDone = False
        scanEvent.set()
        #sc.enableScan()
        lastt=datetime.now()

    # maybe create set of keys in out and remove entries through loop, adding 
    # compliment to log from else after loop
    for k,v in d.items():
        #import pdb;pdb.set_trace()
        v0 = v[0]
        k = scan.getWell(k,sc.pref)

        if out.has_key(k):
            if out[k] != v0:
                f.write("%s current: %s != %s at %s\n"%(k,v0,out[k],scan.strtime()))
                f.flush()
                if updateFromImages:
                    out[k] = v0
            else:
                pass#print v[0]
        else:
            f.write("%s no value setting = %s at %s\n"%(k,v0,scan.strtime()))
            f.flush()
            out[k] = v0
