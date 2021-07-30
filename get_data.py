import pandas as pd

# Dataframe of the categories and their descriptions
variables = pd.DataFrame({
    "Variable Name": ["LoanNr_ChkDgt", "Name", "City", "State", "Zip", "Bank", "BankState",
                        "NAICS", "ApprovalDate", "ApprovalFY", "Term", "NoEmp", "NewExist",
                        "CreateJob", "RetainedJob", "FranchiseCode", "UrbanRural", "RevLineCr",
                        "LowDoc", "ChgOffData", "DisbursementDate", "DisbursementGross",
                        "BalanceGross", "MIS_Status", "ChgOffPrinGr", "GrAppv", "SBA_Appv"],
    "Description": ["Identifier Primary key", "Borrower name", "Borrower city", "Borrower state",
                        "Borrower zip code", "Bank name", "Bank state", "North American industry classification system code",
                        "Date SBA commitment issued", "Fiscal year of commitment", "Loan term in months", "Number of business employees",
                        "1 = Existing business, 2 = New business", "Number of jobs created", "Number of jobs retained",
                        "Franchise code, (00000 or 00001) = No franchise", "1 = Urban, 2 = rural, 0 = undefined",
                        "Revolving line of credit: Y = Yes, N = No", "LowDoc Loan Program: Y = Yes, N = No", 
                        "The date when a loan is declared to be in default", "Disbursement data", "Amount disbursed",
                        "Gross amount outstanding", "Loan status charged off = CHGOFF, Paid in full = PIF",
                        "Charged-off amount", "Gross amount approved by bank","SBA's guaranteed amount of approved loan"]
})


# Read the dataset
df = pd.read_csv("SBAnational.csv", low_memory=False)

# Preprocess Data

# Delete columns that have no statistical significnce (e.g., Loan ID or Borrower's name)
del df["LoanNr_ChkDgt"]
del df["Name"]
del df["City"]
del df["State"]
del df["Zip"]
del df["Bank"]
del df["BankState"]
del df["NAICS"]
del df["ApprovalDate"]
del df["ApprovalFY"]
del df["FranchiseCode"]
del df["ChgOffDate"]
del df["DisbursementDate"]

# Handling Categorical and Text Attributes
df["DisbursementGross"] = df["DisbursementGross"].replace('[\$,]', '', regex=True).astype(float)
df["BalanceGross"] = df["BalanceGross"].replace('[\$,]', '', regex=True).astype(float)
df["ChgOffPrinGr"] = df["ChgOffPrinGr"].replace('[\$,]', '', regex=True).astype(float)
df["GrAppv"] = df["GrAppv"].replace('[\$,]', '', regex=True).astype(float)
df["SBA_Appv"] = df["SBA_Appv"].replace('[\$,]', '', regex=True).astype(float)
df["RevLineCr"] = df["RevLineCr"].apply(lambda x: 1 if x == "Y" else 0)
df["LowDoc"] = df["LowDoc"].apply(lambda x: 1 if x == "Y" else 0)
df["NewExist"] = df["NewExist"].fillna(0)
df["NewExist"] = df["NewExist"].replace(to_replace=0.0, value=2.0)
df["MIS_Status"] = df["MIS_Status"].apply(lambda x: 1 if x == "P I F" else 0)