#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 08:34:05 2023

@author: rodolfofreitas
"""
import numpy as np 
# Compute the conditional mean 
def conditional(x, y, xpts):
    '''
    obtain conditional average of y on x
    
    INPUT:
        x    - independent variable
        y    - dependent variable
        xpts - list of bin points where conditional average is desired (edges of bin)
    OUTPUT:
        xc   - points at which conditional average was taken
        This is the midpoints of (xpts) unless there happens to be          
        no realizations of y in a given bin.  In that case, it is 
        the subset of points where realizations occured.
        yc   - average of y at xc
        ystd - statndard deviation of y at xc
        ymax - maximum value of y at xc
        ymin - minimum value of y at xc
    '''
    n=len(xpts)
    jj=0
    xc = []
    ymean = []
    
    for ii in range(n-1):
        ix = np.where( (x>=xpts[ii]) & (x<=xpts[ii+1]) )
        jj=jj+1
        xc.append( 0.5*(xpts[ii]+xpts[ii+1]))
        ymean.append(np.mean(y[ix]))
            
    return np.stack(xc), np.stack(ymean)