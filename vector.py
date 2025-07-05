from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

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
                Premium: ₹{row["PremiumAmount"]}. Details: {row["Details"]}.
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
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)