import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

def random_gst_number():
    return f"{random.randint(10, 99)}{fake.company_suffix().upper()[:5]}{random.randint(1000, 9999)}Z{random.randint(1, 9)}"

def random_ifsc_code():
    return fake.swift8()[:4] + "0" + str(random.randint(1000, 9999))

def generate_insurance_data(n):
    insurance_types = ['Health', 'Life', 'Vehicle', 'Travel', 'Home']
    data = []
    
    # Create family groups for more realistic data
    family_groups = {}
    for i in range(50):  # Create 50 family groups
        family_id = f"FID-{1000 + i}"
        num_members = random.randint(2, 5)
        
        # Create unique family members with consistent details
        family_members = []
        for j in range(num_members):
            member = {
                'name': fake.unique.name(),
                'dob': fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%Y-%m-%d"),
                'policies': []  # Track policies for this member
            }
            family_members.append(member)
        
        family_groups[family_id] = {
            'members': family_members
        }

    for i in range(n):
        # Select a family and member
        family_id = random.choice(list(family_groups.keys()))
        family_data = family_groups[family_id]
        
        # Select a family member
        member = random.choice(family_data['members'])
        name = member['name']
        dob = member['dob']
        
        policy_id = f"INS-{random.randint(100000, 999999)}"
        policy_number = f"{random.randint(1000000, 9999999)}"
        insurance_type = random.choice(insurance_types)
        issue_date = fake.date_between(start_date='-5y', end_date='today')
        expiry_date = issue_date + timedelta(days=random.randint(365, 1825))
        premium_amount = round(random.uniform(3000, 50000), 2)
        account_number = fake.bban()
        ifsc = random_ifsc_code()
        gst = random_gst_number()
        
        # Generate nominee (usually another family member, sometimes external)
        if random.random() < 0.8:  # 80% chance nominee is family member
            nominee_member = random.choice(family_data['members'])
            nominee_name = nominee_member['name']
        else:  # 20% chance nominee is external
            nominee_name = fake.name()

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
            "FamilyID": family_id,
            "NomineeName": nominee_name,
            "Details": f"{name} holds a {insurance_type} insurance with policy number {policy_number}. It was issued on {issue_date.strftime('%Y-%m-%d')} and is valid till {expiry_date.strftime('%Y-%m-%d')}. The premium is ₹{premium_amount}. Family ID: {family_id}. Nominee: {nominee_name}. IFSC: {ifsc}, GST: {gst}."
        })

    return pd.DataFrame(data)

# Generate and save
df = generate_insurance_data(300)
df.to_csv("insurance_data.csv", index=False)
print("✅ Synthetic insurance data with nominee names saved to insurance_data.csv")
print(f"📊 Generated {len(df)} records with {df['FamilyID'].nunique()} unique families")
print(f"👥 Average policies per family: {len(df) / df['FamilyID'].nunique():.1f}")

# Show sample family data
print("\n📋 Sample Family Data:")
sample_family = df[df['FamilyID'] == 'FID-1001']
if not sample_family.empty:
    print(f"Family FID-1001 members:")
    for _, row in sample_family.iterrows():
        print(f"  - {row['Name']} (DOB: {row['DOB']}) - {row['InsuranceType']}")
