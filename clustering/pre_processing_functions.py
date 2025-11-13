""" 
Cluster data pre-processing module.

This module contains functions used for pre-processing candidate-match data for k-means cluster analysis.

Author: Tom Murray, Candid Hospitality
Date: 16/10/2025
Version:

"""
#----------MODULES-------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

#------NORMALISE-CULTURE-CODES-------------------------------------------------------------------------------------------------

def normalise_culture_code_component(df, attribute_col):
    """
    Normalizes a 1–9 digit personality trait to a scale of [1/9, 1].
    Adds a new column to the DataFrame with the normalized values.
    
    Arguments: 
        df (DataFrame): Input dataframe
        attribute_col (str): Name of column containing attribute to be scaled
        
    Returns: 
        Copy of DataFrame with added column containing normalised attributes
    
    """
    df = df.copy()
    df[f'normalised_{attribute_col}'] = df[attribute_col] / 9
    return df

#-----NORMALISE-SALARIES-------------------------------------------------------------------------------------------------------

def normalise_salary_departmentwise(df, dept_col, salary_col):
    """
    Normalises salaries using z-scores with department means and standard deviations for salaries,
    scale this to range [0,1] using z scores.
    
    Arguments: 
        df (DataFrame): Input dataframe
        dept_col (str): Name of column containing department name, used to find local mean and sd
        salary_col (str): Name of column containing salaries to be normalised
        
    Returns: 
        Copy of DataFrame with added column containing departmentally normalised salaries
    
    """
    df = df.copy()
    
    # convert salries into z scores
    df["z_salary"] = df.groupby(dept_col)[salary_col].transform(
        lambda x: (x-x.mean())/x.std(ddof=0) if x.notna().any() else x
        )
    
    # normalise scores into [0,1]
    df["normalised_salary"] = df.groupby(dept_col)["z_salary"].transform(
        lambda x: ((x-x.min())/(x.max()-x.min())) if x.max() != x.min() else 0.5
        )
    
    # remove intermediate z score column
    return df.drop(columns = "z_salary")

#------NORMALISE-AGES-----------------------------------------------------------------------------------------------------------

def normalise_ages(df, age_col):
    """
    Normalise ages using population mean and standard deviation.
    
    Arguments:
        df (Dataframe): Input dataframe
        age_col (str): Name of column containing ages to be normalised
        
    Returns: 
        Copy of Dataframe with added column containing normalised ages
    
    """
    df = df.copy()
    ages = df[age_col]
    
    min_age = ages.min()
    max_age = ages.max()
    
    # Avoid division by zero if all ages are the same
    if min_age == max_age:
        df["normalised_ages"] = 0.5
    else:
        df["normalised_ages"] = (ages - min_age) / (max_age - min_age)
    
    return df

#-----HAVERSINE-DISTANCE--------------------------------------------------------------------------------------------------------

def haversine_distance(lat1, lng1, lat2, lng2):
    """ 
    Compute "harvesine" distance metric for two coordinates (lat1,lng1), (lat2,lng2) on a sphere.
    
    Arguments: 
        lat1 (float): Latitude coordinate of first point
        lng1 (float): Longitude coordinate of first point
        lat1 (float): Latitude coordinate of second point
        lng1 (float): Longitude coordinate of second point
        
    Returns: Harvesine distance of the two points.
    """
    R = 6371 # Radius of earth
    
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    # harvesine formula
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R*c
    
#-----NEAREST-CITY-------------------------------------------------------------------------------------------------------------

def nearest_city(df, city_col, lat_col, lng_col, cluster_cities):
    """
    Snaps each point in df to the nearest city found in cluster cities list.
    
    Logic:
        - If latitude and longitude unkown, return NaN
        - If city already matches one of desired cities, return city
        - If else, calculate haversine distances between user and cluster cities, then return the city which minimises this
    
    Arguments: 
        df (DataFrame): Input dataframe
        city_col (str): Name of column containing given user city
        lat_col (str): Name of column containing latitude coordinate for user location
        lng_col (str): Name of column containing longitude coordinate for user location
        cluster_cities (DataFrame): Dataframe containing desired cities with latitude and longitude coordinates.
                                   - Must contain columns ["city", "lat", "lng"]
        
    Returns: 
        Copy of DataFrame with added column nearest city
        
    """
    df = df.copy()
    cluster_names = set(cluster_cities['city'].str.lower().str.strip())
    
    def find_nearest(row):
        
        # if latitude and longitude unkown, return na
        if pd.isna(row[lat_col]) or pd.isna(row[lng_col]):
            return np.nan
        
        # if city_name already matches one in cluster_cities, return city_name
        city_name = str(row[city_col]).strip().lower()
        if city_name in cluster_names:
            return row[city_col]
        
        # if else: calculate harvesine distance, return nearest city
        distances = haversine_distance(
            row[lat_col], row[lng_col],
            cluster_cities['lat'].values, cluster_cities['lng'].values
        )
        idx_min = np.argmin(distances)
        return cluster_cities.iloc[idx_min]['city']
    
    df['nearest_city'] = df.apply(find_nearest, axis=1)
    return df

#------CONCAT-USERS-------------------------------------------------------------------------------------------------------------

def concat_users(df, id_col, progressed_col, rejected_col):
    """
    Keep most informative match for users to use for clustering.
    
    Logic:
        For each user id keep one row in following ranking:
        1) If user has progressed in any match use this row as user profile for clustering
        2) If user hasnt progressed, check if they have been rejected. 
           If they have been rejected use this as profile for clustering.
        3) If they have neither progressed or been rejected, use first row as profile.
        Remove other rows with matching user id
        
    Arguments:
        df (DataFrame): Input DataFrame
        id_col (str): Name of column containing user id
        progressed_col (str): Name of column containing progressed label.
                             - Should be 1 if the candidate has progressed, 0 else.
        rejected_col (str): Name of column containing rejected label.
                             - Should be 1 if the candidate was rejected, 0 else.
                             
    Returns:
        Reduced copy of DataFrame containing 1 row per user with most important match info
    """
    df = df.copy()
    
    # assign "priorities"
    df["priority"] = np.where(df[progressed_col] == 1, 2,
                     np.where(df[rejected_col] == 1, 1,
                     0))
    
    # rank by priority
    df = df.sort_values(by=[id_col, 'priority'], ascending=[True, False])
    # keep most important row
    df = df.drop_duplicates(subset=id_col, keep='first')
    
    # remove intermediate priority column 
    return df.drop(columns='priority')

#----ONE-HOT-CITIES------------------------------------------------------------------------------------------------------------

def encode_cities(df, city_col_clean, weight):
    """
    Encode cities for use in clustering.
    
    Arguments: 
        df (DataFrame): Input DataFrame
        city_col_clean(str): Name of column containing city name
                            - IMPORTANT: Function works on column cleaned with nearest_city.
        weight (float): Desired "weight" of binary columns
                            
    Returns:
        DataFrame with added binary column for each city in cluster_cities

    """
    if len(df[city_col_clean].unique()) > 20:
        raise ValueError("Too many cities. Please use nearest_city first.")
    
    # one-hot encode the city column
    encoded = pd.get_dummies(df[city_col_clean], prefix="city")*weight
    
    # join 
    df = pd.concat([df, encoded], axis=1)
    
    return df

#-----ONE-HOT-DEPARTMENTS-------------------------------------------------------------------------------------------------------

def encode_department(df, dept_col, weight):
    """
    Encode department for use in clustering.
    
    Arguments: 
        df (DataFrame): Input DataFrame
        dept_col (str): Name of column containing department name
        weight (float): Desired "weight" of binary columns
                            
    Returns:
        DataFrame with added binary column for each city in cluster_cities

    """
    if len(df[dept_col].unique()) > 20:
        raise ValueError("Too many departments.")
    
    # one-hot encode the department column
    encoded = pd.get_dummies(df[dept_col], prefix="department")*weight
    
    # join 
    df = pd.concat([df, encoded], axis=1)
    
    return df

#----MASTER-FUNCTION-----------------------------------------------------------------------------------------------------------

def cluster_preprocessing(df,
                          id_col, progressed_col, rejected_col, 
                          city_col, lat_col, lng_col, cluster_cities, 
                          dept_col, salary_col, age_col, culture_cols =[], 
                          weighting_cols = [], weight =0.2):
    """
    Master function for cluster pre_processsing.
    
    Arguments: 
        df (DataFrame): Input Dataframe
        id_col (str): Name of column containing user id
        progressed_col (str): Name of column containing progressed label
        rejected_col (str): Name of column containing rejected label
        city_col (str): Name of column containing city
        lat_col (str): Name of column containing user latitude coordinate
        lng_col (str): Name of column containing user longitude coordinate
        cluster_cities (DataFrame): DataFrame of cluster cities, must contain ["city", "lat", "lng"]
        dept_col (str): Name of column containing user department
        salary_col (str): Name of column containing user expected salary
        age_col (str): Name of column containing user age
        culture_cols (list): List of attribute columns (1-9 scale)
        weighting_cols (list): List of binary columns to be weighted
        weight (float): Desired "weight" of columns in weighting_cols
        
    Returns:
        Preprocessed DataFrame ready for clustering

    """
    df = df.copy()
    
    # 1) keep most informative match per user
    df = concat_users(df, id_col, progressed_col, rejected_col)
    
    # 2) snap to nearest city
    df = nearest_city(df, city_col, lat_col, lng_col, cluster_cities)
    
    # 3) encode nearest city
    df = encode_cities(df, 'nearest_city', weight)
    
    # 4) encode department
    df = encode_department(df, dept_col, weight)
    
    # 5) normalise salaries departmentally
    df = normalise_salary_departmentwise(df, dept_col, salary_col)
    
    # 6) normalize ages
    df = normalise_ages(df, age_col)
    
    # 7) normalize culture code components
    for col in culture_cols:
        df = normalise_culture_code_component(df, col)
    
    return df

    
    
                                       
        
        
        
    
    
        
    
