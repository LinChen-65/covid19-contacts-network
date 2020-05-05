# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:16:40 2020

@author: Michelle
"""
import pandas as pd
import numpy as np
import math
from math import sin,radians,cos,asin,sqrt
from itertools import combinations
import h5py 

min_dis=5
EARTH_REDIUS = 6378.137
t_window=30
days=31
max_pointnum=int(days*24*60/t_window)
path ='/home/stu/wangmudan/'

def contact_Matrix(t_tra):
    combins = [c for c in combinations(t_tra,2)] #排列组合
    combins= pd.DataFrame(combins)
    combins['dis']=combins.apply(lambda x:getDistance(x[0],x[1]),axis=1)
    contacts=combins[combins['dis']<=min_dis]
    contacts=[(c1[0],c2[0]) for c1,c2 in zip(list(combins[0]),list(combins[1]))]
    return contacts 

def rad(d):
    pi=3.141592654
    return d*pi/180.0

def getDistance(x1,x2):
    """
    gps1 = (39.982628,116.220291)
    gps2 = (39.98718344444445,116.23183122222223)""" 

    gps1=(x1[1],x1[2])
    gps2=(x2[1],x2[2])
    
    lat1=gps1[0]
    lng1=gps1[1]
    lat2=gps2[0]
    lng2=gps2[1]
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(sin(a/2), 2) + cos(radLat1) * cos(radLat2) * math.pow(sin(b/2), 2)))
    s = s * EARTH_REDIUS
    return s*1000

def main():
    with h5py.File(path+"Interpolation.hdf5",'r') as f:  
        all_tra = f.get("tra_Interpolation_30min_test")[:]
    uid_n=all_tra[:,-1]
    uid=uid_n[:,0]
    all_time_contacts=[]
    for i in range(max_pointnum):
        t_tra=all_tra[:,i+1]  #t_tra也是numpy.array
        t_tra[:,0]=uid
        contacts=contact_Matrix(t_tra)
        all_time_contacts.append(contacts)
    
    #保存文件
    f=h5py.File(path+"contacts.hdf5","w")
    f.create_dataset("all_time_contacts",data=np.array(all_time_contacts))   

if __name__=="__main__":
    main()
    print('complete..')

    



 
    

