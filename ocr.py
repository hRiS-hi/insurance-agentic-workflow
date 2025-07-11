import pdfplumber
import pandas as pd
import re

def clean_text(text):
    return text.strip() if text else ""

def extract_policy_info(pdf_path):
    data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            
            # Debug: Print first few lines to see the format
            print(f"Processing PDF: {pdf_path}")
            print("First 10 lines of extracted text:")
            for i, line in enumerate(text.split('\n')[:10]):
                print(f"Line {i+1}: {line}")
            print("-" * 50)
            
            # Initialize a dictionary with all required fields including nominee name only
            policy_data = {
                'PolicyID': '',
                'Name': '',
                'DOB': '',
                'PolicyNumber': '',
                'InsuranceType': '',
                'IssueDate': '',
                'ExpiryDate': '',
                'PremiumAmount': '',
                'AccountNumber': '',
                'IFSCCode': '',
                'GSTNumber': '',
                'FamilyID': '',
                'NomineeName': '',
                'Details': ''
            }
            
            # Extract information using patterns
            lines = text.split('\n')
            details = []
            in_details = False
            
            # Debug: Track what we're finding
            found_fields = []
            
            for line in text.split('\n'):
                # PolicyID - more flexible
                policy_id_match = re.search(r'PolicyID:\s*(POL-\d+)', line, re.IGNORECASE)
                if policy_id_match:
                    policy_data['PolicyID'] = clean_text(policy_id_match.group(1))
                    found_fields.append('PolicyID')
                else:
                    # Try alternative patterns
                    policy_id_match = re.search(r'Policy\s*ID:\s*(POL-\d+)', line, re.IGNORECASE)
                    if policy_id_match:
                        policy_data['PolicyID'] = clean_text(policy_id_match.group(1))
                        found_fields.append('PolicyID')
                
                # Name - more flexible
                name_match = re.search(r'Name:\s*([^$\n]+)', line, re.IGNORECASE)
                if name_match:
                    policy_data['Name'] = clean_text(name_match.group(1))
                    found_fields.append('Name')
                else:
                    # Try alternative patterns
                    name_match = re.search(r'Policyholder:\s*([^$\n]+)', line, re.IGNORECASE)
                    if name_match:
                        policy_data['Name'] = clean_text(name_match.group(1))
                        found_fields.append('Name')
                    else:
                        name_match = re.search(r'Insured:\s*([^$\n]+)', line, re.IGNORECASE)
                        if name_match:
                            policy_data['Name'] = clean_text(name_match.group(1))
                            found_fields.append('Name')
                
                # DOB
                dob_match = re.search(r'DOB:\s*(\d{2}/\d{2}/\d{4})', line)
                if dob_match:
                    policy_data['DOB'] = clean_text(dob_match.group(1))
                
                # PolicyNumber
                policy_num_match = re.search(r'PolicyNumber:\s*(HINS-\d{4}-\d+)', line)
                if policy_num_match:
                    policy_data['PolicyNumber'] = clean_text(policy_num_match.group(1))
                
                # InsuranceType
                insurance_type_match = re.search(r'InsuranceType:\s*([^$\n]+(?:Insurance|insurance))', line)
                if insurance_type_match:
                    policy_data['InsuranceType'] = clean_text(insurance_type_match.group(1))
                
                # IssueDate
                issue_date_match = re.search(r'IssueDate:\s*(\d{2}/\d{2}/\d{4})', line)
                if issue_date_match:
                    policy_data['IssueDate'] = clean_text(issue_date_match.group(1))
                
                # ExpiryDate
                expiry_date_match = re.search(r'ExpiryDate:\s*(\d{2}/\d{2}/\d{4})', line)
                if expiry_date_match:
                    policy_data['ExpiryDate'] = clean_text(expiry_date_match.group(1))
                
                # PremiumAmount
                premium_match = re.search(r'PremiumAmount:\s*\$?([\d,]+(?:\.\d{2})?)', line)
                if premium_match:
                    policy_data['PremiumAmount'] = clean_text(premium_match.group(1))
                
                # AccountNumber
                account_match = re.search(r'AccountNumber:\s*(\d{8})', line)
                if account_match:
                    policy_data['AccountNumber'] = clean_text(account_match.group(1))
                
                # IFSCCode
                ifsc_match = re.search(r'IFSCCode:\s*([A-Z]{4}0\d{6})', line)
                if ifsc_match:
                    policy_data['IFSCCode'] = clean_text(ifsc_match.group(1))
                
                # GSTNumber
                gst_match = re.search(r'GSTNumber:\s*(\d{2}AABCT\d{4}1Z\d)', line)
                if gst_match:
                    policy_data['GSTNumber'] = clean_text(gst_match.group(1))
                
                # FamilyID - more flexible pattern
                family_id_match = re.search(r'FamilyID:\s*(FID-\d+)', line, re.IGNORECASE)
                if family_id_match:
                    policy_data['FamilyID'] = clean_text(family_id_match.group(1))
                    found_fields.append('FamilyID')
                else:
                    # Try alternative patterns
                    family_id_match = re.search(r'Family\s*ID:\s*(FID-\d+)', line, re.IGNORECASE)
                    if family_id_match:
                        policy_data['FamilyID'] = clean_text(family_id_match.group(1))
                        found_fields.append('FamilyID')
                    else:
                        family_id_match = re.search(r'Family\s*:\s*(FID-\d+)', line, re.IGNORECASE)
                        if family_id_match:
                            policy_data['FamilyID'] = clean_text(family_id_match.group(1))
                            found_fields.append('FamilyID')
                
                # NomineeName - more flexible pattern
                nominee_name_match = re.search(r'NomineeName:\s*([^$\n]+)', line, re.IGNORECASE)
                if nominee_name_match:
                    policy_data['NomineeName'] = clean_text(nominee_name_match.group(1))
                    found_fields.append('NomineeName')
                else:
                    # Try alternative patterns
                    nominee_name_match = re.search(r'Nominee\s*Name:\s*([^$\n]+)', line, re.IGNORECASE)
                    if nominee_name_match:
                        policy_data['NomineeName'] = clean_text(nominee_name_match.group(1))
                        found_fields.append('NomineeName')
                    else:
                        nominee_name_match = re.search(r'Nominee:\s*([^$\n]+)', line, re.IGNORECASE)
                        if nominee_name_match:
                            policy_data['NomineeName'] = clean_text(nominee_name_match.group(1))
                            found_fields.append('NomineeName')
                        else:
                            nominee_name_match = re.search(r'Beneficiary:\s*([^$\n]+)', line, re.IGNORECASE)
                            if nominee_name_match:
                                policy_data['NomineeName'] = clean_text(nominee_name_match.group(1))
                                found_fields.append('NomineeName')
                
                # Collect Details/Terms and Conditions
                if 'Terms and Conditions:' in line:
                    in_details = True
                    continue
                elif in_details and line.strip():
                    details.append(line.strip())
            
            # Join all details into one string
            if details:
                policy_data['Details'] = ' '.join(details)
            
            # Debug: Show what fields were found
            print(f"Fields found: {found_fields}")
            print(f"FamilyID: '{policy_data['FamilyID']}'")
            print(f"NomineeName: '{policy_data['NomineeName']}'")
            print("-" * 30)
            
            # Only add if we have essential information
            if policy_data['PolicyNumber'] or policy_data['PolicyID']:
                data.append(policy_data)
    
    return data

def main():
    # List of PDF files to process
    import glob
    pdf_files = glob.glob('fake_insurance_policy_*.pdf')
    
    if not pdf_files:
        print("No insurance policy PDFs found in the current directory.")
        return
    
    all_data = []
    
    # Process each PDF
    for pdf_file in pdf_files:
        try:
            pdf_data = extract_policy_info(pdf_file)
            all_data.extend(pdf_data)
            print(f"Successfully processed {pdf_file}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
    # Create DataFrame and save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Ensure columns are in the specified order including nominee name only
        columns = ['PolicyID', 'Name', 'DOB', 'PolicyNumber', 'InsuranceType', 
                  'IssueDate', 'ExpiryDate', 'PremiumAmount', 'AccountNumber', 
                  'IFSCCode', 'GSTNumber', 'FamilyID', 'NomineeName', 'Details']
        
        df = df[columns]  # Reorder columns
        
        csv_path = 'extracted_insurance_data.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nData successfully saved to {csv_path}")
        
        # Display the first few rows of extracted data
        print("\nExtracted Data Preview:")
        print(df.head())
        
        # Display some statistics
        print("\nExtraction Statistics:")
        print(f"Total policies processed: {len(df)}")
        print("Fields extracted per policy:")
        for column in columns:
            filled = df[column].astype(bool).sum()
            print(f"{column}: {filled}/{len(df)} ({filled/len(df)*100:.1f}%)")
    else:
        print("No data was extracted from the PDFs")

if __name__ == "__main__":
    main()