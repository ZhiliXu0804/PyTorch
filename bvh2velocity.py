import math 
import numpy as np

def butter_worth(inData, deltaTimeInSec, cutOff):
    data = np.copy(inData)
    if (not data) or (not cutOff): 
        return data
    dF2 = len(inData) - 1  # The data range is set with dF2
    Dat2 = np.zeros(dF2 + 4) # Array with 4 extra points front and back
	# Copy inData to Dat2
    Dat2[2:dF2+2] = inData[:]
    Dat2[1] = Dat2[0] = inData[0]
    Dat2[dF2 + 3] = Dat2[dF2 + 2] = inData[dF2]
    
    wc = math.tan(cutOff * math.pi * deltaTimeInSec)
    k1 = math.sqrt(2) * wc # Sqrt(2) * wc
    k2 = wc * wc
    a = k2 / (1 + k1 + k2)
    b = 2 * a
    c = a
    k3 = b / k2
    d = -2 * a + k3
    e = 1 - (2 * a) - k3
    # RECURSIVE TRIGGERS - ENABLE filter is performed (first, last points constant)
    DatYt = np.zeros(dF2 + 4)
    DatYt[1] = DatYt[0] = inData[0]
    for s in range(2,dF2+2):
          DatYt[s] = a * Dat2[s] + b * Dat2[s - 1] + c * Dat2[s - 2] + d * DatYt[s - 1] + e * DatYt[s - 2]
    DatYt[dF2 + 3] = DatYt[dF2 + 2] = DatYt[dF2 + 1]

	# FORWARD filter
    DatZt = np.zeros(dF2 + 2)
    DatZt[dF2] = DatYt[dF2 + 2]
    DatZt[dF2 + 1] = DatYt[dF2 + 3]
    for t in range(-dF2 + 1, 1): 
        DatZt[-t] = a * DatYt[-t + 2] + b * DatYt[-t + 3] + c * DatYt[-t + 4] + d * DatZt[-t + 1] + e * DatZt[-t + 2]
    
    data[:dF2 + 1] = DatZt[:]

    return data
