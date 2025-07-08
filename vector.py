from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import re

df = pd.read_csv("insurance_data.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content = f"""
                Name: {row["Name"]}, born on {row["DOB"]}, holds policy number {row["PolicyNumber"]} ({row["InsuranceType"]} insurance), issued on {row["IssueDate"]} and expiring on {row["ExpiryDate"]}.
                Premium: ₹{row["PremiumAmount"]}, FamilyID: {row["FamilyID"]}, Nominee: {row["NomineeName"]}, Details: {row["Details"]}.
                """,

            metadata={
                "PolicyID": row["PolicyID"],
                "Name": row["Name"],
                "DOB": row["DOB"],
                "PolicyNumber": row["PolicyNumber"],
                "InsuranceType": row["InsuranceType"],
                "IssueDate": row["IssueDate"],
                "ExpiryDate": row["ExpiryDate"],
                "PremiumAmount": row["PremiumAmount"],
                "AccountNumber": row["AccountNumber"],
                "IFSCCode": row["IFSCCode"],
                "GSTNumber": row["GSTNumber"],
                "FamilyID": row["FamilyID"],
                "NomineeName": row["NomineeName"],
                "Details": row["Details"]
            },
          
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="insurance_data",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

def is_family_query(query):
    """Check if the query is asking about family insurance"""
    family_keywords = [
        'family', 'families', 'family insurance', 'family members', 
        'family policies', 'family coverage', 'family plan',
        'my family', 'our family', 'family details', 'family information',
        'family members insurance', 'family insurance details'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in family_keywords)

def get_family_members_by_name(name, df):
    """Get all family members when given a person's name"""
    # First, find the person's FamilyID
    person_record = df[df['Name'].str.lower() == name.lower()]
    if person_record.empty:
        return []
    
    family_id = person_record.iloc[0]['FamilyID']
    
    # Get all members with the same FamilyID
    family_members = df[df['FamilyID'] == family_id]
    return family_members.to_dict('records')

def get_family_members_by_family_id(family_id, df):
    """Get all family members by FamilyID"""
    family_members = df[df['FamilyID'] == family_id]
    return family_members.to_dict('records')

def get_family_summary(family_id, df):
    """Get a summary of family insurance information"""
    family_members = get_family_members_by_family_id(family_id, df)
    
    if not family_members:
        return None
    
    summary = {
        'family_id': family_id,
        'total_members': len(family_members),
        'total_premium': sum(float(member['PremiumAmount']) for member in family_members),
        'insurance_types': list(set(member['InsuranceType'] for member in family_members)),
        'members': []
    }
    
    for member in family_members:
        member_info = {
            'name': member['Name'],
            'insurance_type': member['InsuranceType'],
            'policy_number': member['PolicyNumber'],
            'premium': member['PremiumAmount'],
            'expiry_date': member['ExpiryDate'],
            'nominee': member['NomineeName']
        }
        summary['members'].append(member_info)
    
    return summary

def enhanced_retriever(query):
    """Enhanced retriever that handles family insurance queries"""
    # Check if this is a family-related query
    if is_family_query(query):
        # First, try to find any names mentioned in the query
        # Look for common name patterns (more comprehensive)
        name_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b',  # First Middle Last
            r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # First M. Last
            r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z]\.\b',  # First Last M.
        ]
        
        names_found = []
        for pattern in name_patterns:
            names_found.extend(re.findall(pattern, query))
        
        if names_found:
            # If names are found, get family members for the first name
            family_members = get_family_members_by_name(names_found[0], df)
            if family_members:
                # Convert family members to documents
                documents = []
                for member in family_members:
                    doc = Document(
                        page_content=f"""
                            Name: {member["Name"]}, born on {member["DOB"]}, holds policy number {member["PolicyNumber"]} ({member["InsuranceType"]} insurance), issued on {member["IssueDate"]} and expiring on {member["ExpiryDate"]}.
                            Premium: ₹{member["PremiumAmount"]}, FamilyID: {member["FamilyID"]}, Nominee: {member["NomineeName"]}, Details: {member["Details"]}.
                            """,
                        metadata=member
                    )
                    documents.append(doc)
                return documents
        
        # If no specific names found, try to find FamilyID in the query
        family_id_pattern = r'FID-\d+'
        family_ids_found = re.findall(family_id_pattern, query)
        
        if family_ids_found:
            # Get family members by FamilyID
            family_members = get_family_members_by_family_id(family_ids_found[0], df)
            if family_members:
                documents = []
                for member in family_members:
                    doc = Document(
                        page_content=f"""
                            Name: {member["Name"]}, born on {member["DOB"]}, holds policy number {member["PolicyNumber"]} ({member["InsuranceType"]} insurance), issued on {member["IssueDate"]} and expiring on {member["ExpiryDate"]}.
                            Premium: ₹{member["PremiumAmount"]}, FamilyID: {member["FamilyID"]}, Nominee: {member["NomineeName"]}, Details: {member["Details"]}.
                            """,
                        metadata=member
                    )
                    documents.append(doc)
                return documents
        
        # If no specific names or FamilyID found, do a general search and group by FamilyID
        # Get all documents from vector store
        all_docs = vector_store.get()
        
        # Group by FamilyID
        family_groups = {}
        for i, metadata in enumerate(all_docs['metadatas']):
            if metadata and 'FamilyID' in metadata:
                family_id = metadata['FamilyID']
                if family_id not in family_groups:
                    family_groups[family_id] = []
                family_groups[family_id].append(all_docs['documents'][i])
        
        # Return documents from the first family group (or you could return multiple)
        if family_groups:
            first_family = list(family_groups.values())[0]
            return [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in first_family]
    
    # For non-family queries, use the regular retriever
    return vector_store.as_retriever(search_kwargs={"k": 10}).invoke(query)

# Create the regular retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Create the enhanced retriever
enhanced_retriever_instance = enhanced_retriever