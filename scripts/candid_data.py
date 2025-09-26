import pandas as pd
import candid_cleaning as cl

# Read unclean data
users_raw = pd.read_csv("C:\\Users\\tommu\\Downloads\\New folder\\Candid Hospitality\\augmenting data cleaning\\users.csv")
matches_raw = pd.read_csv("C:\\Users\\tommu\\Downloads\\New folder\\Candid Hospitality\\augmenting data cleaning\\wppp_matches (3).csv")
chats_raw = pd.read_csv("C:\\Users\\tommu\\Downloads\\New folder\\Candid Hospitality\\augmenting data cleaning\\wppp_chats (13).csv")

# Split off bios for sentiment analysis 
bios_raw = users_raw[["user_id", "culture_text"]].copy()

# List of test bios for removal
test_accounts = [21,18,31,39,43,44,94,104,114,115,113,391,440,507,
                 578,579,591,605,607,654,808,1532,3991,4387,4116,
                 4117,4118,4327,4328,4329,4460,4462,4463,711,2613,5331,5685]

# Apply formulas to user data
users_raw["dob"] = users_raw["dob"].apply(cl.clean_dobs)
users_raw[["risk","extroversion","patience","norms"]] = users_raw["culture_code"].apply(cl.split_cc)
users_raw["age"] = cl.dob_to_age(users_raw["dob"])
users_raw = cl.remove_accounts(users_raw, test_accounts, id_col = "user_id")

# Apply formulas to chats
chat_summary = cl.chat_stats(chats_raw, time_unit="m")

# Apply formulas to carry out sentiment analysis
sentiment_scores = bios_raw["culture_text"].apply(cl.bio_sentiment_analysis)
sentiment_df = pd.json_normalize(sentiment_scores)
bios_raw = pd.concat([bios_raw, sentiment_df], axis=1)

# Apply formulas to match data 
matches_raw["liked"] = matches_raw["liked"].apply(cl.date_to_binary)
matches_raw["disliked"] = matches_raw["disliked"].apply(cl.date_to_binary)
matches_raw["progressed"] = matches_raw["progressed"].apply(cl.date_to_binary)
matches_raw["rejected"] = matches_raw["rejected"].apply(cl.date_to_binary)

# Rename match column in matches for consistency 
matches_raw = matches_raw.rename(columns={'id': 'match_id'})
matches_raw = matches_raw.rename(columns={'candidate_id': 'user_id'})

# Merge match and chat data
match_chat = pd.merge(matches_raw, chat_summary, on='match_id', how='left')

# Merge chat/match data with user data 
user_match_chat = pd.merge(match_chat, users_raw, on='user_id', how='left')

# Merge chat/match/user data with bio sentiment data 
full_data = pd.merge(user_match_chat, bios_raw, on='user_id', how='left')

# Choose columns to subset
subset_cols = ["job_id" , "user_id" , "score_overall" , "score_department" , "score_culture" , "score_competencies" , "score_compensation",
               "score_benefits" , "liked" , "disliked" , "progressed" , "rejected" , "candidate_msgs" , "company_msgs" , "candidate_response_time" , 
               "company_response_time" , "interactivity_metric" , "ethnicity" , "gender" , "current_city" , "department_name" , "expected_salary" , 
               "risk" , "extroversion" , "patience" , "norms" , "age" , "neg" , "pos" , "compound"]

# Create smaller more usable dataframe, rename columns for clarity
candid_data = full_data[subset_cols].copy()
candid_data = candid_data.rename(columns={ "neg":"bio_sentiment_neg" , "pos":"bio_sentiment_pos" , "compound":"bio_sentiment_compound"})

