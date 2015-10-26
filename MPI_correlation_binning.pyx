'''
__license__   = "GNU GPLv3 <https://www.gnu.org/licenses/gpl.txt>"
__copyright__ = "2015, Joseph Kuruvilla"
__author__    = "Joseph Kuruvilla <joseph.k@uni-bonn.de>"
__version__   = "2.0"

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Changelog:

v2.0: Added arguements to calculate ensemble average for x and y axes as line of sight in addition to z axis.
'''

# --------------------
# Importing modules
# --------------------

from __future__ import division
import numpy as np
cimport numpy as np

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t

DTYPEi = np.int64
ctypedef  np.int64_t DTYPEi_t

DTYPEu = np.uint8
ctypedef np.uint8_t DTYPEu_t

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

# --------------------
# Defining functions
# --------------------


def twod_ensemble(np.ndarray[DTYPEf_t, ndim=3] data, int res, int indexl, int indexu, int innerl, int inneru, float radius1, float radius2, float height1, float height2, int axis):  
    '''
    Function which will iterate and return ensemble average for fixed radius for 2-D.
    Output is in the form of rperp1, rpara1
			     rperp1, rpara2 ...
			     
    Axis arguements should be: 1 - xaxis, 2 - yaxis, 3 - zaxis 
    '''
    cdef float sum_mask = 0
    cdef int i,j,k
    cdef np.ndarray[DTYPEi_t, ndim=3] x
    cdef np.ndarray[DTYPEi_t, ndim=3] y
    cdef np.ndarray[DTYPEi_t, ndim=3] z
    cdef np.ndarray[DTYPEu_t, ndim=3, cast=True] R
   
    for i in xrange(indexl,indexu):
      for j in xrange(innerl, inneru):
        for k in xrange(res):
          x,y,z = np.ogrid[-i:(res-i),-j:(res-j),-k:(res-k)]
          for l in xrange(res):
            if x[l,0,0] > res//2:
              x[l,0,0] = x[l,0,0] - res
            if y[0,l,0] > res//2:
              y[0,l,0] = y[0,l,0] - res
            if z[0,0,l] > res//2:
              z[0,0,l] = z[0,0,l] - res
          
          if axis == 1:
            R = np.logical_and(np.logical_or(np.logical_and(x>height1,x<=height2), np.logical_and(x<-height1,x>=-height2)), np.logical_and(z**2 + y**2<= radius2**2, z**2 + y**2 > radius1**2))
          elif axis == 2:
            R = np.logical_and(np.logical_or(np.logical_and(y>height1,y<=height2), np.logical_and(y<-height1,y>=-height2)), np.logical_and(x**2 + z**2<= radius2**2, x**2 + z**2 > radius1**2))
          elif axis == 3:
            R = np.logical_and(np.logical_or(np.logical_and(z>height1,z<=height2), np.logical_and(z<-height1,z>=-height2)), np.logical_and(x**2 + y**2<= radius2**2, x**2 + y**2 > radius1**2))
          
          sum_mask += (data[i,j,k] * np.average(data[R]))
   
    return sum_mask


def oned_ensemble(np.ndarray[DTYPEf_t, ndim=3] data, int res, int indexl, int indexu, int innerl, int inneru, float radius1, float radius2):  
    '''
    Function which will iterate and return correlation for fixed radiusfor 1-D.
    '''
    cdef float sum_mask = 0
    cdef int i,j,k, l
    cdef int a, b, c
    cdef np.ndarray[DTYPEi_t, ndim=3] x
    cdef np.ndarray[DTYPEi_t, ndim=3] y
    cdef np.ndarray[DTYPEi_t, ndim=3] z
    cdef np.ndarray[DTYPEu_t, ndim=3, cast=True] R
   
    for i in xrange(indexl,indexu):
      for j in xrange(innerl, inneru):
        for k in xrange(res):
          a,b,c = res//2,res//2,res//2
          x,y,z = np.ogrid[-i:(res-i),-j:(res-j),-k:(res-k)]
          for l in xrange(res):
            if x[l,0,0] > res//2:
              x[l,0,0] = x[l,0,0] - res
            if y[0,l,0] > res//2:
              y[0,l,0] = y[0,l,0] - res
            if z[0,0,l] > res//2:
              z[0,0,l] = z[0,0,l] - res
                    
          R = np.logical_and(x**2 + y**2 + z**2 <= radius2**2, x**2 + y**2 + z**2 > radius1**2)
          sum_mask += (data[i,j,k] * np.average(data[R]))
   
    return sum_mask


def exact(np.ndarray[DTYPEf_t, ndim=3] data, np.ndarray[DTYPEf_t, ndim=3] vel_data, int res, int indexl, int indexu, int innerl, int inneru, float radius1, float radius2, float height1, float height2, float kpara, int axis):  
    '''
    Function used to compute the exact redshift space power spectrum. Function will iterate over each element 
    and return correlation for fixed radius.
    
    eg. exact(combined_array, velz, 32, 0, 1, rperp[4], rperp[5], output, rpara[4], rpara[5], kpara[1])
    '''
    cdef float sum_mask = 0
    cdef float counter = 0
    cdef int i,j,k, 
    cdef np.ndarray[DTYPEi_t, ndim=3] x
    cdef np.ndarray[DTYPEi_t, ndim=3] y
    cdef np.ndarray[DTYPEi_t, ndim=3] z
    cdef np.ndarray[DTYPEu_t, ndim=3, cast=True] R

    for i in xrange(indexl,indexu):
      for j in xrange(innerl, inneru):
        for k in xrange(res):
          x,y,z = np.ogrid[-i:(res-i),-j:(res-j),-k:(res-k)]
          for l in xrange(res):
            if x[l,0,0] > res//2:
              x[l,0,0] = x[l,0,0] - res
            if y[0,l,0] > res//2:
              y[0,l,0] = y[0,l,0] - res
            if z[0,0,l] > res//2:
              z[0,0,l] = z[0,0,l] - res
              
          if axis == 1:
            R = np.logical_and(np.logical_or(np.logical_and(x>height1,x<=height2), np.logical_and(x<-height1,x>=-height2)), np.logical_and(z**2 + y**2<= radius2**2, z**2 + y**2 > radius1**2))
          elif axis == 2:
            R = np.logical_and(np.logical_or(np.logical_and(y>height1,y<=height2), np.logical_and(y<-height1,y>=-height2)), np.logical_and(x**2 + z**2<= radius2**2, x**2 + z**2 > radius1**2))
          elif axis == 3:
            R = np.logical_and(np.logical_or(np.logical_and(z>height1,z<=height2), np.logical_and(z<-height1,z>=-height2)), np.logical_and(x**2 + y**2<= radius2**2, x**2 + y**2 > radius1**2))
                              
          den_data = data[R]
          v_data = vel_data[R]
          counter = 0
          for ii in range(len(den_data)):
            counter += (np.cos(kpara*0.4955441023414177*(vel_data[i,j,k] - v_data[ii]))*(data[i,j,k]*den_data[ii]))
          counter /= len(den_data)
          sum_mask += counter
    
    return sum_mask

def total(np.ndarray[DTYPEf_t, ndim=3] data, np.ndarray[DTYPEf_t, ndim=3] vel_data, int res, int indexl, int indexu, int innerl, int inneru, float radius1, float radius2, float height1, float height2, float kpara, int axis):  
    '''
    Function used to compute the total redshift space power spectrum. Total implies the Taylor expansion of the cos function in the 
    exact formula. Funciton will iterate and return correlation for fixed radius
    
    eg. total(combined_array, velz, 32, 0, 1, rperp[4], rperp[5], output, rpara[4], rpara[5], kpara[1])
    '''
    cdef float sum_mask = 0
    cdef float counter = 0
    cdef int i,j,k, 
    cdef np.ndarray[DTYPEi_t, ndim=3] x
    cdef np.ndarray[DTYPEi_t, ndim=3] y
    cdef np.ndarray[DTYPEi_t, ndim=3] z
    cdef np.ndarray[DTYPEu_t, ndim=3, cast=True] R

    for i in xrange(indexl,indexu):
      for j in xrange(innerl, inneru):
        for k in xrange(res):
          x,y,z = np.ogrid[-i:(res-i),-j:(res-j),-k:(res-k)]
          for l in xrange(res):
            if x[l,0,0] > res//2:
              x[l,0,0] = x[l,0,0] - res
            if y[0,l,0] > res//2:
              y[0,l,0] = y[0,l,0] - res
            if z[0,0,l] > res//2:
              z[0,0,l] = z[0,0,l] - res
              
          if axis == 1:
            R = np.logical_and(np.logical_or(np.logical_and(x>height1,x<=height2), np.logical_and(x<-height1,x>=-height2)), np.logical_and(z**2 + y**2<= radius2**2, z**2 + y**2 > radius1**2))
          elif axis == 2:
            R = np.logical_and(np.logical_or(np.logical_and(y>height1,y<=height2), np.logical_and(y<-height1,y>=-height2)), np.logical_and(x**2 + z**2<= radius2**2, x**2 + z**2 > radius1**2))
          elif axis == 3:
            R = np.logical_and(np.logical_or(np.logical_and(z>height1,z<=height2), np.logical_and(z<-height1,z>=-height2)), np.logical_and(x**2 + y**2<= radius2**2, x**2 + y**2 > radius1**2))
                        
          den_data = data[R]
          v_data = vel_data[R]
          counter = 0
          for ii in range(len(den_data)):
            counter += ((1-(0.5*(kpara**2)*0.24556395736536146*(vel_data[i,j,k] - v_data[ii])**2))*(data[i,j,k]*den_data[ii]))
          counter /= len(den_data)
          sum_mask += counter
    
    return sum_mask


def vel_diff(np.ndarray[DTYPEf_t, ndim=3] data, int res, int indexl, int indexu, int innerl, int inneru, float radius1, float radius2, float height1, float height2, int axis):  
    '''
    Function will calculate the ensemble average of the velocity difference for a given radius. Function will iterate over each points
    and return velocity difference for fixed radius.
    
    eg. total(velz, 32, 0, 1, rperp[4], rperp[5], output, rpara[4], rpara[5])
    '''
    cdef float sum_mask = 0
    cdef float counter = 0
    cdef int i,j,k, 
    cdef np.ndarray[DTYPEi_t, ndim=3] x
    cdef np.ndarray[DTYPEi_t, ndim=3] y
    cdef np.ndarray[DTYPEi_t, ndim=3] z
    cdef np.ndarray[DTYPEu_t, ndim=3, cast=True] R
   
    for i in xrange(indexl,indexu):
      for j in xrange(innerl, inneru):
        for k in xrange(res):
          x,y,z = np.ogrid[-i:(res-i),-j:(res-j),-k:(res-k)]
          for l in xrange(res):
            if x[l,0,0] > res//2:
                    x[l,0,0] = x[l,0,0] - res
            if y[0,l,0] > res//2:
                    y[0,l,0] = y[0,l,0] - res
            if z[0,0,l] > res//2:
                    z[0,0,l] = z[0,0,l] - res
                    
          if axis == 1:
            R = np.logical_and(np.logical_or(np.logical_and(x>height1,x<=height2), np.logical_and(x<-height1,x>=-height2)), np.logical_and(z**2 + y**2<= radius2**2, z**2 + y**2 > radius1**2))
          elif axis == 2:
            R = np.logical_and(np.logical_or(np.logical_and(y>height1,y<=height2), np.logical_and(y<-height1,y>=-height2)), np.logical_and(x**2 + z**2<= radius2**2, x**2 + z**2 > radius1**2))
          elif axis == 3:
            R = np.logical_and(np.logical_or(np.logical_and(z>height1,z<=height2), np.logical_and(z<-height1,z>=-height2)), np.logical_and(x**2 + y**2<= radius2**2, x**2 + y**2 > radius1**2))
                              
          v_data = data[R]
          counter = 0
          for ii in range(len(v_data)):
            counter += (data[i,j,k] - v_data[ii])**2
          counter /= len(v_data)
          sum_mask += counter
   
    return sum_mask
