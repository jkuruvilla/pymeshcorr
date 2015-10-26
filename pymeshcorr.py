'''
__license__   = "GNU GPLv3 <https://www.gnu.org/licenses/gpl.txt>"
__copyright__ = "2015, Joseph Kuruvilla"
__author__    = "Joseph Kuruvilla <joseph.k@uni-bonn.de>"
__version__   = "1.0"

Program to determine the correlation function in the given grid of size 'n x n x n' using MPI4py

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
'''

# ------------------
# Importing Modules
# ------------------

import sys
import numpy as np
import struct
from mpi4py import MPI
import MPI_correlation_binning as cor
  
# ----------------------------
# Global variable declaration
#-----------------------------

scale_factor   = 1 #parameters of the simulation
omegam, omegal = 0.279, 0.721 #parameters of the simulation
H              = 100*np.sqrt((omegam/scale_factor**3)+omegal)
f              = omegam**0.55
tes_res        = int(sys.argv[1]) #int(raw_input("Enter the resolution of the tessellated grid: "))
box_size       =   #enter the box size here
ax             = int(sys.argv[2]) #1 - xaxis, 2 - yaxis, 3 - zaxis
  
# -------------------
# Defining Functions
# -------------------

def read_data(file, n):
    '''
    Function to read the file which is in the struct format and change to float format. Further we reshape the array to the 3D structure.
    '''    
    with open(file, "rb") as infile:
      fil=infile.read()
    l_den=np.array(struct.unpack("f" * ((len(fil) ) // 4), fil[:]))    ## struct is used to unpack
    av_tes_data = np.average(l_den)
    den_con = np.array([((x/av_tes_data)-1) for x in l_den])
    den_con = den_con.reshape(n,n,n)
    print "Reading of data done"
    return den_con

def read_noavgdata_div(file, n):
    '''
    Function to read the file which is in the struct format and change to float format. Further we reshape the array to the 3D structure.
    '''
    with open(file, "rb") as infile:
      fil=infile.read()
    l_den=np.array(struct.unpack("f" * ((len(fil) ) // 4), fil[:]))    ## struct is used to unpack    
    if ax == 1:
      l_den = l_den[0::9]
    elif ax == 2:
      l_den = l_den[4::9]
    elif ax == 3:
      l_den = l_den[8::9]
    l_den = -l_den[:]/(scale_factor*H)
    l_den = l_den.reshape(tes_res,tes_res,tes_res)
    return l_den    


def read_noavgdata_vel(file, n):
    '''
    Function to read the file which is in the struct format and change to float format. Further we reshape the array to the 3D structure.
    '''
    with open(file, "rb") as infile:
      fil=infile.read()
    l_den=np.array(struct.unpack("f" * ((len(fil) ) // 4), fil[:]))    ## struct is used to unpack    
    if ax == 1:
      l_den = l_den[0::3]
    elif ax == 2:
      l_den = l_den[1::3]
    elif ax == 3:
      l_den = l_den[2::3]
    l_den = -l_den[:]/(f*scale_factor*H)
    l_den = l_den.reshape(tes_res,tes_res,tes_res)
    return l_den    


# --------------------
#   Program  start
# --------------------    

if __name__ == "__main__":
   
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  
  if tes_res == 64:
    r = np.linspace(0, 8, 7)
  elif tes_res == 128:
    r = np.linspace(0, 14, 11)
  elif tes_res == 256:
    r = np.linspace(0, 28, 16)

  if ax == 1:
    ff = 'x'
  elif ax == 2:
    ff = 'y'
  elif ax == 3:
    ff = 'z'
    
  input_file = "" #raw_input("Enter the density file: ") 
  den_contrast = read_data(input_file, tes_res)  if comm.rank == 0  else None #density contrast file is read 
  """
  input_file = "" #raw_input("Enter the gradient file: ")
  divergence = read_noavgdata_div(input_file, tes_res) if comm.rank == 0  else None  #velocity gradient file is read
   
  combined_array = (den_contrast + divergence) if comm.rank == 0  else None
  
  
  input_file = ""  #raw_input("Enter the velocity file: ")
  velocity = read_noavgdata_vel(input_file, tes_res) if comm.rank == 0  else None

  
  input_file = "" #raw_input("Enter the density file: ") 
  den_contrast = read_data(input_file, tes_res)  if comm.rank == 0  else None #density contrast file is read 
  """
 
  data = comm.bcast(den_contrast, root=0)
  #data1 = comm.bcast(velocity, root=0)  

  corr_2d = []
  a = np.zeros(1)
  intt = np.zeros(1)
   
  subdivision = 4 # 4 subdvisions here are 128^3 grids to parallelise using 512 cores

  if (rank%subdivision) == 0: 
    aa = (rank/subdivision)
    bb = aa + 1
    cc = 0
    dd = 32 

  elif (rank%subdivision) == 1:
    aa = (rank/subdivision)
    bb = aa + 1
    cc = 32 
    dd = 64 

  elif (rank%subdivision) == 2:
    aa = (rank/subdivision)
    bb = aa + 1
    cc = 64 
    dd = 96 

  elif (rank%subdivision) == 3:
    aa = (rank/subdivision)
    bb = aa + 1
    cc = 96 
    dd = 128 
  
   
  k = 0
  while k < (len(r)-1):
    l = 0
    while l < (len(r)-1):
      intt[0] = cor.twod_ensemble(data, tes_res, aa, bb, cc, dd, r[k], r[k+1], r[l], r[l+1], ax)
      comm.Reduce(intt, a, op = MPI.SUM, root = 0)
      
      if comm.rank == 0:
        corr_2d.append(a/float(tes_res**3))
        #print(a/float(tes_res**3))
      l+=1
    k+=1
  
  """ 
  kpara = np.linspace(0.006, 0.125, 15)
  mm = 0
  full_corr2d = []
  while mm < (len(kpara)):
    corr_2d = []
    k = 0
    while k < (len(r)-1):
      l = 0
      while l < (len(r)-1):
        intt[0] = cor.exact(data, data1, tes_res, aa, bb, cc, dd, r[k], r[k+1], r[l], r[l+1], kpara[mm], ax)
        comm.Reduce(intt, a, op = MPI.SUM, root = 0)
      
        if comm.rank == 0:
          corr_2d.append(a/float(tes_res**3))
        #print(a/float(tes_res**3))
        l+=1
      k+=1
    mm+=1
    full_corr2d.append(corr_2d)
  """

  file = open("", "w")
  
  rr = []
  for j in range(len(r)-1):
    rr.append(((r[j]+r[j+1])/2)*(box_size/float(tes_res)))
  
  rperp = []
  for i in range(len(r)-1):
    for j in range(len(r)-1):
      rperp.append(rr[i])

  rpara = []
  for i in range(len(r)-1):
    for j in range(len(r)-1):
      rpara.append(rr[j])
  
   
  for i in range(len(corr_2d)):
    file.write(str(rperp[i])+"\t"+str(rpara[i])+"\t"+str(corr_2d[i])+"\n")
  file.close()   
  
  """
  for j in range(len(kpara)):
    for i in range(len(corr_2d)):
      file.write(str(kpara[j])+"\t"+str(rperp[i])+"\t"+str(rpara[i])+"\t"+str(full_corr2d[j][i])+"\n") 
  file.close()  
  """  
