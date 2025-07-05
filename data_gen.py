import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

def random_gst_number():
    return f"{random.randint(10, 99)}{fake.company_suffix().upper()[:5]}{random.randint(1000, 9999)}Z{random.randint(1, 9)}"

def random_ifsc_code():
    return fake.swift8()[:4] + "0" + str(random.randint(1000, 9999))

def generate_insurance_data(n=100):
    insurance_types = ['Health', 'Life', 'Vehicle', 'Travel', 'Home']
    data = []

    for i in range(n):
        name = fake.name()
        dob = fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%Y-%m-%d")
        policy_id = f"INS-{random.randint(100000, 999999)}"
        policy_number = f"{random.randint(1000000, 9999999)}"
        insurance_type = random.choice(insurance_types)
        issue_date = fake.date_between(start_date='-5y', end_date='today')
        expiry_date = issue_date + timedelta(days=random.randint(365, 1825))
        premium_amount = round(random.uniform(3000, 50000), 2)
        account_number = fake.bban()
        ifsc = random_ifsc_code()
        gst = random_gst_number()

        data.append({
            "PolicyID": policy_id,
            "Name": name,
            "DOB": dob,
            "PolicyNumber": policy_number,
            "InsuranceType": insurance_type,
            "IssueDate": issue_date.strftime("%Y-%m-%d"),
            "ExpiryDate": expiry_date.strftime("%Y-%m-%d"),
            "PremiumAmount": premium_amount,
            "AccountNumber": account_number,
            "IFSCCode": ifsc,
            "GSTNumber": gst,
            "Details": f"{name} holds a {insurance_type} insurance with policy number {policy_number}. It was issued on {issue_date.strftime('%Y-%m-%d')} and is valid till {expiry_date.strftime('%Y-%m-%d')}. The premium is ₹{premium_amount}. IFSC: {ifsc}, GST: {gst}."
        })

    return pd.DataFrame(data)

# Generate and save
df = generate_insurance_data(100)
df.to_csv("insurance_data.csv", index=False)
print("✅ Synthetic insurance data saved to insurance_data.csv")
