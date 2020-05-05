# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:11:19 2020
@author: Michelle
"""
# -*- coding: utf-8 -*-
import numpy as np
import collections
import h5py 
import pandas as pd
t_window=30
days=31
max_pointnum=int(days*24*60/t_window)
weekend=[5,6,12,13,19,20,26,27]
day_interval=int(24*60/t_window)
LON1 = 115.7
LON2 = 117.4
LAT1 = 34.0
LAT2 = 41.6

path='/home/stu/wangmudan/filter_tra_thu/'
def read_from_text(line):
    yield line.strip('\r\n').split('\t')

'''1. Processing data'''
#==============================================================================
#map timestamp to timeID
def MaptimeID(trace_list):
    #1. 2018-01-01 00:00:00(1514736000)-2018-01-31 24:00:00 (1517414400)
    # 10 minutes: a time slot 
    start=1514736000 
    trace=[]
    for i in range(0,len(trace_list)): 
        timestamp=trace_list[i].split(',')[0]
        timeID=int((int(timestamp)-start)/60/t_window)  
        lat=trace_list[i].split(',')[2]
        lon=trace_list[i].split(',')[1]
        trace.append([int(timeID),round(float(lat),5),round(float(lon),5)])         
    return np.array(trace)

# sort timeID
def Sort_timeID(trace):
    trace=np.array(trace)
    idex=np.lexsort([trace[:,0]])
    sorted_trace = trace[idex,:]        
    return sorted_trace

#Delete duplicate timeID
def dropDu(tra):
    d_tra=[]
    d_tra.append(tra[0])
    for i in range(1,len(tra)):
        time1=tra[i-1][0]
        time2=tra[i][0]
        if time2!=time1:
            d_tra.append(tra[i])
    return np.array(d_tra)     

#Delete trajectories records that do not meet the criteria
def tra_Filter(dropDu_trace):
    if len(dropDu_trace)>10:
        day=dropDu_trace[:,0]/day_interval
        day=day.astype(int)
        day=np.unique(day)
        #记录大于10天的情况
        if len(day)>10:
            return 1
        else:
            return -1
    else:
        return -1
    
# add missing timeID
def padding(dropTra):
    dropTra=dropTra.tolist()
    new_trace=[]
    t=0
    for k in range(max_pointnum):
        if t<len(dropTra):
            if k==int(dropTra[t][0]):
                new_trace.append(dropTra[t])
                t+=1 
            else:
                new_trace.append([float(k),-1.0,-1.0])
        else:
            new_trace.append([float(k),-1.0,-1.0])
    return new_trace

'''2. probability_Interpolation '''
#============================================================================== 
def recognize_weekday(new_trace):
    day_tra=[] #工作日
    end_tra=[] #周末
    for i in range(0,len(new_trace)):
        day=int(i/day_interval)
        if day in weekend:
            end_tra.append(new_trace[i])
        else:
            day_tra.append(new_trace[i])
    return np.array(day_tra),np.array(end_tra)

def gridsID(lon,lat,column_num,row_num):
    if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
        return -1
    column = (LON2-LON1)/column_num
    row = (LAT2-LAT1)/row_num
    return int((lon-LON1)/column)+ 1 + int((lat-LAT1)/row) * column_num

def MapGrids(trace):
    '''
    map lnglat to grids and map grid to regionID
    Beijing: 115.7°—117.4°,39.4°—41.6°
    146232.36229347554 846028.1301393477
    ''' 
    column_num=1000
    row_num=10000
    lon=trace[:,2]
    lat=trace[:,1]
    timeID=trace[:,0]
  
    c={'timeID':timeID,
       "lon":lon,
       'lat':lat} 
    
    grid= pd.DataFrame(c)
    grid['gridID'] = grid.apply(lambda x: gridsID(x['lon'],x['lat'],column_num,row_num),axis = 1)
    grid_tra=grid[['timeID','lat','lon','gridID']].values.tolist()
    return np.array(grid_tra)


def Counter(freq,flag,grid_tra):
    if len(freq)!=0:
        counter=collections.Counter(freq)
        values=np.array(list(counter.values()))
        pro_list=values/values.sum()
        v_P=max(pro_list)
        if v_P>=0.4:
            key=list(counter.keys())[list(pro_list).index(v_P)]
            if flag=='grid':
                traKey=grid_tra[grid_tra[:,3]==key,:]
                lat=traKey[:,1].mean() 
                lon=traKey[:,2].mean()
                gps=str((lat,lon)) 
            else:
                gps=key
            return gps
        else:
            return -1 

def pro_compute(max_gps,m,flag,tra):
    for i in range(day_interval):
        freq=[]
        for j in range(i,len(tra),day_interval):
            if max_gps[i]=='':
                if flag=='grid':
                    if tra[i][3]!=-1:
                        freq.append(tra[i][3])       
                else:
                    if tra[i][1]!=-1:
                        freq.append(str((tra[i][1],tra[i][2])))
        if len(freq)>m:  
            gps=Counter(freq,flag,tra)
            if gps!=-1:
                max_gps[i]=gps
    return max_gps

def get_max_gps(tra,m):
    max_gps=['' for i in range(day_interval)]
    max_gps=pro_compute(max_gps,m,'gps_point',tra)
    if '' in max_gps:
        grid_tra=MapGrids(tra)
        max_gps=pro_compute(max_gps,m,'grid',grid_tra)
    return max_gps

        
def probability_Interpolation(l,u,tra,max_gps1,max_gps2):
    for j in range(l,u):
        t=j%day_interval
        day=int(j/day_interval)
        #如果是工作日 取出对应的gps；如果是周末 取出对应的gps；
        if day in weekend:
            if max_gps2[t]!='':
                in_gps=eval(max_gps2[t])
                tra[j][1]=in_gps[0]
                tra[j][2]=in_gps[1]
        else:
            if max_gps1[t]!='':
                in_gps=eval(max_gps1[t])
                tra[j][1]=in_gps[0]
                tra[j][2]=in_gps[1]
    return tra 

'''3. Linear_Interpolation'''

#==============================================================================
def Linear_Interpolation(time1,time2,loc1,loc2,tra,missing_num):
    
    if loc1[0] == loc2[0] and loc1[1] == loc2[1]:
        for i in range(1,missing_num):
            tra[time1+i][1]=loc1[0]
            tra[time1+i][2]=loc1[1]
    else:
        for i in range(1,missing_num):
            lat = loc1[0]+(loc2[0]-loc1[0])*i/(missing_num+1)
            lon=loc1[1]+(loc2[1]-loc1[1])*i/(missing_num+1)
            tra[time1+i][1]=lat
            tra[time1+i][2]=lon
    return tra

#==============================================================================        
def Interpolation(tra,dropTra,max_gps1,max_gps2,f):

    t_threshold=t_window/t_window*4 #时间间隔的阈值
    t_init=0
    num=max_pointnum-1

    # 1. add missing points before the first record
    t0=int(dropTra[0][0])
    if t0-t_init>=t_threshold and f=='first':#循环概率插值 
        tra=probability_Interpolation(t_init,t0,tra,max_gps1,max_gps2)
    else:
        for i in range(t_init,t0):
            tra[i][1]=dropTra[0][1]
            tra[i][2]=dropTra[0][2]
    
    # 2. add missing points in the blank time of the trajectory by using interpolation
    for i in range(1,len(dropTra)):
        t1=int(dropTra[i-1][0])
        t2=int(dropTra[i][0])

        if t2-t1>=t_threshold and f=='first':
            tra=probability_Interpolation(t1,t2,tra,max_gps1,max_gps2)
        else:
            loc1=(dropTra[i-1][1],dropTra[i-1][2])
            loc2=(dropTra[i][1],dropTra[i][2])
            missing_num=int(t2-t1)
            tra=Linear_Interpolation(t1,t2,loc1,loc2,tra,missing_num)
    
    # 3. add missing points after the last record
    t_end=int(dropTra[-1][0])
    if t_end!=num:
        if num-t_end>=t_threshold and f=='first':
            tra=probability_Interpolation(t_end,num,tra,max_gps1,max_gps2)
        else:
            for i in range(t_end,num+1):
                tra[i][1]=dropTra[-1][1]
                tra[i][2]=dropTra[-1][2]
    return tra      

#==============================================================================
def main():
    allTrace=[]
    fileLines=open(path+'filter_tra_thu','r').readlines()
    for j,lines in enumerate(fileLines):
        for line in read_from_text(lines):		
            uid=int(line[0])
            trace=line[1]
            trace_list=trace.split(';')  
            trace=MaptimeID(trace_list)
            if len(trace)!=0:
                sortTra=Sort_timeID(trace)
                dropTra=dropDu(sortTra)
                signal=tra_Filter(dropTra)
                if signal==-1:
                    continue
                new_trace=padding(dropTra)
                day_tra,end_tra=recognize_weekday(new_trace)  
                max_gps1=get_max_gps(day_tra,10) 
                max_gps2=get_max_gps(end_tra,4)
                new_tra=Interpolation(new_trace,dropTra,max_gps1,max_gps2,'first')
                new_tra=Interpolation(new_trace,dropTra,max_gps1,max_gps2,'second')

                uid=[[uid,0.0,0.0]]
                u=np.concatenate((new_tra,uid),axis=0) 
                allTrace.append(u)            

    f=h5py.File(path+"Interpolation.hdf5","w")
    f.create_dataset("tra_Interpolation_30min_test",data=np.array(allTrace))

       
if __name__ == "__main__":
    main()
    print('complete...')
    '''
    步骤：
    1、删除总记录len<10;删除没有周期性的数据（例如在某一天的一个时间段有一些记录，其他天都缺失）
    2、将工作日和周末分别进行插值 （如果不分开的话 大概率在工作时间点估计在办公室，但对于周末是不成立的）
    3、工作日（23天）：
        if 两个相邻的time 间隔>2个小时：
           if 在t时刻，23天gps点的记录len>10 并且某个点的概率>0.5---->在缺失值的time 插入该点的轨迹
           else: 网格化.
               if 在t时刻，23天网格的记录len>10 并且某个网格的概率>0.5---->在缺失值的time 插入网格中心点的轨迹
               else:线性插值
        else:
            线性插值
    4、周末（8天）：
        和工作日类似
    '''  

    
      