import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer

#------------------------------------------------------------------------------------------------------------------------

# DOB cleaning function
def clean_dobs(dob):
    """
    -Fixes incorrectly formatted DOBs, converts YYYYMMDD to YYYY-MM-DD
    
    Arguments:
        dob: raw series of DOBs
        
    Returns: 
        dob_str: a corrcetd series of DOBs, NaNs still remain, other errors are fixed
    """
    # If DOB is nan return without changing
    if pd.isna(dob):
        return dob
    
    # Convert DOB to string 
    dob_str = str(dob)
    
    # If YYYY-MM-DD, return 
    if "-" in dob_str and len(dob_str) == 10:
        return dob_str
    
    # If YYYYMMDD, split and add '--'
    if dob_str.isdigit() and len(dob_str) == 8:
        return f"{dob_str[:4]}-{dob_str[4:6]}-{dob_str[6:]}"
    
    # Any other case, return original
    return dob_str 

#---------------------------------------------------------------------------------------------------------------------

# split culture codes function
def split_cc(cc):
    """
    Culture codes should be 4 digits,some are 5 digits or na (error).
    Each digit is a seperate "score".
    - Checks if a code is na, if so returns na for each component
    - If not na, convert codes(stored as floats) to integers
    - Convert integers to strings to allow spliting
    - Split codes into sepeate components
    - If a code is 5 digits, rteurn na for each component
    
    Arguments:
        cc: raw series of culture codes
        
    Returns:
        Risk, Extroversion, Patience, Norms
        If code is ABCD, Risk = A, Extroversion = B, Patience = C, Norms = D
        If code is ABCDE, Risk = nan, Extroversion = nan, Patience = nan, Norms = nan
        If code is nan, Risk = nan, Extroversion = nan, Patience = nan, Norms = nan
    
    """
    # If code is nan, return 4 length series of nan
    if pd.isna(cc):
        return pd.Series([np.nan]*4, index=["risk","extroversion","patience","norms"])
    
    # Convert float to integer, then to string
    if isinstance(cc, float) and cc.is_integer():
        cc_str = str(int(cc))
    else:
        cc_str = str(cc).strip()
    
    # Split valid code into 4 components
    if len(cc_str) == 4 and cc_str.isdigit():
        return pd.Series([int(cc_str[0]), int(cc_str[1]), int(cc_str[2]), int(cc_str[3])],
                         index=["risk","extroversion","patience","norms"])
    
    # Split 5 digit erroneous code into 4 length series of nan
    if len(cc_str) == 5 and cc_str.isdigit():
        return pd.Series([np.nan]*4, index=["risk","extroversion","patience","norms"])
    
    # Any other case, return 4 length series of nan
    return pd.Series([np.nan]*4, index=["risk","extroversion","patience","norms"])  

#-----------------------------------------------------------------------------------------------------------------------------

# calculate age function
def dob_to_age(dob_col):
    """
    Calculates age of user using the (cleaned) DOB column.
    - Convert dob column to age
    
    Arguments: 
        dob_col: CLEANED column of DOBs
        
    Returns:
        age: column of user ages 
    """
    dob = pd.to_datetime(dob_col, errors="coerce")
    today = pd.to_datetime("today").normalize()
    
    # Age calculation using years
    age = today.year - dob.dt.year
    
    # -1 if birthday hasn't occurred yet this year
    before_birthday = (today.month < dob.dt.month) | (
        (today.month == dob.dt.month) & (today.day < dob.dt.day)
    )
    age -= before_birthday.astype(int)
    
    return age

#------------------------------------------------------------------------------------------------------------------------------

# function to work out chat "stats"
def chat_stats(chat_df, time_unit="s"):
    """
    Count the number of chats for candidate and company per match,
    compute average response time for candidate and company,
    and calculate interaction metric.
    
    Arguments:
        chat_df: contains id, match_id, sender,timestamp
        time_unit: "s" = seconds, "m" = minute, "h" = hours
        
    Returns:
        chat_stats: contains match_id, candidate_messages, company_messages, 
        candidate_response_time, company_response_time, interactivity_metric
    """
    # Keep only candidate and company messages, remove system messages
    df = chat_df[chat_df["sender"].isin(["candidate", "company"])].copy()
    # Make sure timestamp is stores as datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Count messages for both candidate and company per match
    counts = (
        df.groupby(["match_id", "sender"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"candidate": "candidate_msgs", "company": "company_msgs"})
    )
    
    # Sort chats into match id's by timestamp, earliest message first
    df = df.sort_values(["match_id", "timestamp"])
    df["next_sender"] = df.groupby("match_id")["sender"].shift(-1)
    df["next_time"] = df.groupby("match_id")["timestamp"].shift(-1)
    
    # Only consider when sender changes
    mask = df["sender"] != df["next_sender"]
    responses = df.loc[mask, ["match_id", "sender", "timestamp", "next_sender", "next_time"]].copy()
    
    # Calculate time difference
    responses["response_time"] = (responses["next_time"] - responses["timestamp"]).dt.total_seconds()
    
    # Attribute response time to the responder (the "next_sender")
    responses = (
        responses.groupby(["match_id", "next_sender"])["response_time"]
        .mean()
        .unstack()
        .rename(columns={"candidate": "candidate_response_time",
                         "company": "company_response_time"})
    )
    
    # Convert units if needed
    if time_unit == "m":  # minutes
        responses = responses / 60
    elif time_unit == "h":  # hours
        responses = responses / 3600
    
    # Merge accounts and reponse times
    chat_stats = counts.merge(responses, on="match_id", how="left").reset_index()
    
    # Calculate "interactivity metric"
    #          -> 1 candidate dominates
    #          -> 0 company dominates
    chat_stats["interactivity_metric"] = (
        (chat_stats["candidate_msgs"] - chat_stats["company_msgs"]) /
        (chat_stats["candidate_msgs"] + chat_stats["company_msgs"])
    )
    
    return chat_stats

#----------------------------------------------------------------------------------------------------------------------------

#function to convert date columns to binary
def date_to_binary(date):
    """ 
    Convert "date" variables (liked, disliked, progressed, rejected) to binary so they are machine-readable.
    """
    if pd.isna(date):
        return 0
    else:
        return 1
    
#----------------------------------------------------------------------------------------------------------------------------

#function to run bio sentiemnet analysis

sia = SentimentIntensityAnalyzer()

def bio_sentiment_analysis(bio):
    """
    Run sentiment analysis on users bios, store this to use for predictive modelling.
    
    Arguments:
        bio: series of user bios
        
    Return: 
        sia.polarity_scores: rating of negative, neutral, positive, overall content in user bios

    """
    
    
    if pd.isna(bio):  # handles NaN values
        return {"neg": 0, "neu": 0, "pos": 0, "compound": 0}
    
    bio = str(bio)
    
    
    return sia.polarity_scores(bio)

#--------------------------------------------------------------------------------------

#function to remove test accounts

def remove_accounts(users, tests, id_col = "user_id"):
    """
    Removes test accounts from users data
    
    Arguments:
        users: raw series of all user ids
        tests: list of ids of test accounts
        
    Returns:
        list of user ids with test accounts removed
    """
    users = users[~users[id_col].isin(tests)].copy()
    return users
    