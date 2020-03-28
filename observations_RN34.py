#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 08:47:41 2019

@author: eke001
"""
import numpy as np
import models


import porepy as pp


def read_pressure_temperature(files):
    """ Read data from xls files.
    
    Parameter:
        str, or list of str: List of files to be read.
    
    Returns:
        dictionary of Pandas data frames. The keys are drawn from the headers
            in the xls files, these are typically a date. Added to this is a 
            pressure or temperature identifier (drawn from the file name).
            The dictionary value is a pandas data frame of two columns, the 
            first is observation depth, the second is observed value.
        
    """
    import pandas as pd
    data = {}
    if not isinstance(files, list):
        files = [files]
    for f in files:

        is_pressure = '_P' in f
        is_temperature = '_T' in f

        df = pd.read_excel(f)
        for col in range(1, df.shape[1]):
            sub_df = df.iloc[:, [0, col]].dropna()

            name = sub_df.columns.values[1].split(' ')[0]
            if is_pressure:
                key = 'pressure_' + name
            elif is_temperature:
                key = 'temperature_' + name
            else:
                raise ValueError('Unknown data field in file ' + f)
            data[key] = sub_df

    return data


def load_pressure_observations_march_29():
    
    # First, read off every 2.5 minutes
    observed_pressure_0950_1022 = np.array([128,  # 09:50
                                               122, 119, 116, 113.4,  # 10:00
                                               112, 110.5, 109.5, 108.3,  # 10:10
                                               107.3, 106.5, 105.9, 105.5,  # 10:20
                                               105.0])
    # Next, every 5 minutes
    observed_pressure_1025_1120 = np.array([104.7, 104, 103.5, 103.1,  # 10:40
                                        102.6, 102.2, 101.8, 101.4, # 11:00
                                        101.0, 100.7, 100.4, 100.2,  # 11:20
                                        99.9, 99.6, 99.3, 99.0, # 11:40
                                        98.8, 98.5, 98.3, 98.0, # 12:00
                                        97.8, 97.6, 97.4, 97.3, # 12:20
                                        ])
    mean_p = 0.5 * (observed_pressure_1025_1120[:-1] + observed_pressure_1025_1120[1:] )
    
    # Insert mean values for pressures so that the observations are equi-spaced in time
    expanded = np.zeros(observed_pressure_1025_1120.size + mean_p.size)
    expanded[::2] = observed_pressure_1025_1120
    expanded[1::2] = mean_p
    
    
    
    
    observed_pressure = np.hstack((observed_pressure_0950_1022, expanded))
    
    observation_time_first = 2.5 * pp.MINUTE * np.arange(observed_pressure_0950_1022.size)
    observation_time_second = 5 * pp.MINUTE * (1 + np.arange(observed_pressure_1025_1120.size - 1))
    observation_time_second = np.hstack((observation_time_second, observation_time_second[-1] + 2.5 * pp.MINUTE))
    # Offset from end of previous period
    observation_time_second += observation_time_first[-1]

    #observed_time = np.hstack((observation_time_first, observation_time_second))    
    
    # The pressure values are now given every 2.5 minutes
    observed_time = 2.5 * pp.MINUTE * np.arange(observed_pressure.size)
    
    # Mimic the fall off test: The injection was reported to last from 7:15 to 9:50
    # (interpreted from p30 and p100 in ISOR report 2015-017). The pressure buildup 
    # during this test is mainly unknown, the only available information is that the
    # pressure was more or less constont over the last 30 minutes of the test 
    # (see Figure 46 in the ISOR report)
    # 
    dt = 150
    start_injection = -2 * pp.HOUR + 35 * pp.MINUTE
#    known_plateau = -30 * pp.MINUTE
    t_prior = np.arange(start_injection, 0, dt)
    p_prior = observed_pressure[-1] + np.power(t_prior - t_prior[0], 0.4) * 1.03
    
    if False:  # Use this to create figure
        t = np.hstack((observed_time[0] - 30 * pp.MINUTE, observed_time[:-20]))
        p = np.hstack((observed_pressure[0], observed_pressure[:-20]))
    else:
        t = np.hstack((t_prior, observed_time))
        p = np.hstack((p_prior, observed_pressure))
    
    if False:  # Activate this to create figures
        import datetime
        import matplotlib.pyplot as plt
        
        t0 = datetime.datetime(year=2015, month=3, day=29, hour=9, minute=50, second=0)
        
        t_full = [t0 + datetime.timedelta(seconds=i) for i in t]
        plt.plot(t_full, p, linewidth=5)
        fig = plt.gcf()
        fig.autofmt_xdate()
        # ax = fig.gca()
    #    for tick in ax.xaxis.get_major_ticks():
    #        tick.set_fontsize(16)
    #    for tick in ax.yaxis.get_major_ticks():
    #        tick.set_fontsize(16)        
            
        plt.ylabel('Pressure [bar]', fontsize=24)
        plt.show()
        
        t0_inj = datetime.datetime(year=2015, month=3, day=29, hour=12, minute=00, second=0)
        dt = datetime.timedelta(minutes=30)
        q_inj_cycle = [100, 100, 100, 20, 20]
        t_inj_cycle = [t0_inj + i * dt for i in [0, 1, 2, 2, 3]]
            
        q_inj = [q_inj_cycle[j] for i in range(7) for j in range(len(q_inj_cycle))]
        t_inj = [3*i *dt  + t_inj_cycle[j] for i in range(7) for j in range(len(t_inj_cycle))]
        
        t_inj = t_inj[:-1]
        q_inj = q_inj[:-1]
        
        plt.figure()
        fig = plt.gcf()
        plt.plot(t_inj, q_inj, linewidth=5)
        fig.autofmt_xdate()
        plt.ylabel('Injection rate [L/s]', fontsize=24)    
    
        plt.show()
    
    return t, p
    #return inj_rate, length_periods, observed_pressure, observed_time, num_data_points_periods
    

def load_pressure_observations_april_7():
    """ Injection rates and lengths of injection periods. 
     Measured out of 52 in ISOR report 2015-017.
    """
    inj_rates = np.array([40, 60, 20])
    length_periods = np.array([3.25, 3, 3]) * pp.HOUR

    ## Observed_pressure in period 13:07:30 - 17:00. 7.5 minutes between data points
    observed_pressure_until_17 = np.array([146.8, # 13:07:30
                                  149, 150.5, 151, # 13:30
                                  151.3, 151.8, 152.2, 152.6, # 14:00
                                  152.8, 153, 153.3, 153.5, # 14:30
                                  153.8, 154.1, 154.3, 154.5, # 15:00
                                  154.7, 154.8, 154.9, 155.1, # 15:30
                                  155.3, 155.4, 155.7, 155.8, # 16:00
                                  155.9, 156.0, 156.0]) # 16:22:30
                                  
    observation_time_first = 7.5 * pp.MINUTE * np.arange(observed_pressure_until_17.size)
    
    ## Observations during second injection stage. 15 minutes between data points.
    # The main reasons for longer time between observations are laziness + it seems to be sufficient
    observed_pressure_1715_1922 = np.array([159.2, # 16:30
                                            161.7, 163.2,  # 17:00 
                                            164.2, 165.1, 165.8, 166.5,  # 18:00
                                            167.0, 167.7, 168.1, 168.5,  # 19:00
                                            168.9, # 19:15
                                            169.2  # 19:22:30
                                            ])
    # Observation times, will be added to first group (thus start arange on 1)
    observation_time_second = 15 * pp.MINUTE * (1 + np.arange(observed_pressure_1715_1922.size - 1))
    # Add a final observation at about 19:22:30
    observation_time_second = np.hstack((observation_time_second, observation_time_second[-1] + 7.5 * pp.MINUTE))
    # Offset from end of previous period
    observation_time_second += observation_time_first[-1]
    
    observed_pressure_1930_2230 = np.array([165, 163, 161.2, 160, 159.6, # 20:00
                                            159.2, 158.6, 158.1, 157.8,  # 20:30
                                            157.6, 157.2, 156.9, 156.7, # 21:00
                                            156.5, 156.3, 156.1, 155.9, # 21:30
                                            155.7, 155.5, 155.4,155.3, # 22:00
                                            155.1, 155, 154.8 # 22:22:30
                                            ])
    
    # Observation times, will be added to second group (thus start arange on 1)
    observation_time_third = 7.5 * pp.MINUTE * (1 + np.arange(observed_pressure_1930_2230.size))
    # Offset from end of second period
    observation_time_third += observation_time_second[-1]
    
    # Use time-averaging of the first and third periods, as these are denser, thus
    # more prone to round-off errors
    observed_pressure_until_17[1:-1] = 0.5 * (observed_pressure_until_17[:-2] + observed_pressure_until_17[2:])
    observed_pressure_1930_2230[1:-1] = 0.5 * (observed_pressure_1930_2230[:-2] + observed_pressure_1930_2230[2:])
    
    
    observed_pressure = np.hstack((observed_pressure_until_17, observed_pressure_1715_1922, observed_pressure_1930_2230))
    
    observed_time = np.hstack((observation_time_first, observation_time_second, observation_time_third))

    num_data_points_periods = np.array([observation_time_first.size, observation_time_second.size, observation_time_third.size])

    return inj_rates, length_periods, observed_pressure, observed_time, num_data_points_periods

def read_seismic_locations(domain_center=None):
    
    # Coordinates of the locations, transformed to isnet93 coordinates.
    # To do the conversion, put the UTM coordinates in a file and run
    #  $ cs2cs +init=epsg:32627 +to +init=epsg:3057 infile -r > outfile
    # NBNB: The -r option is used to switch between xy and yx ordering
    xy_file = '/home/eke001/Dropbox/workspace/prosjekter/Eris/seismic_observations/seismic_locations_xy_isnet93.data'
    
    xy = np.genfromtxt(xy_file)[:, :2]
    
    # File with raw data, we'll pull the depths of the location from here
    depth_file = '/home/eke001/Dropbox/workspace/prosjekter/Eris/seismic_observations/seismic_event_location.txt'
    # Skip one header file, and use the 6th coordinate (assumed to be double difference depths)
    z = np.genfromtxt(depth_file, skip_header=1)[:,6].reshape((-1, 1))
    
    # negative depth
    z *= -1
    
    if True or domain_center is None:
        model = models.GeometryData()
        domain_center = model.get_domain_center()
        
    xy -= domain_center[:2]
    locations = np.hstack((xy, z))
        
    return locations

if __name__ == '__main__':
    load_pressure_observations_march_29()