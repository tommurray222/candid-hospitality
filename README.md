# candid-hospitality
Collection of python code and data set snippets used during my data internship at **Candid Hospitality**, an anonymous matchmaking service for hospitality careers.
The focus of this project was cleaning and preparing the data for analysis, before moving on to prediction.

## Scripts

- **candid_cleaning** is a collection of functions used to correct errors in the data, and fix any formatting issues the data has when importing.
#### Dependencies: pandas, numpy, nltk.sentiment

- **candid_data** Script for loading, transforming and preparing the (test) data for analysis.
#### Dependencies: pandas, candid_cleaning (must run this script first to load functions)

## Datasets

**Due to the personal nature of the data, all datasets have been anonymised and have had all indicators removed**
**For this reason all birthdates have been moved to XXXX-01-01, presrving only the birth year for age analysis**

- **users_snippet.csv** – User data including age, department, city, culture code, expected salary, and user ID.
- **matches_snippet.csv** – Matches between jobs and users with various scoring metrics (`score_overall`, `score_culture`, `score_competencies`, `score_compensation`, `score_benefits`) and status flags (`liked`, `disliked`, `progressed`, `rejected`).
- **chats_snippet.csv** – Sample chat data used for analysis

**Disclaimer**: Datasets are intended for demonstration and testing use only, and are not representative of the actual candiate pool at **Candid Hospitality**.


