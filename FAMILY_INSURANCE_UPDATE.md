# Enhanced Family Insurance Retrieval System

## Overview

The insurance chatbot has been enhanced to provide comprehensive family insurance information retrieval. When users ask about family insurance, the system now automatically retrieves information about all family members connected through the same FamilyID.

## Key Features

### 1. Family Query Detection
The system automatically detects when a user is asking about family insurance using keywords such as:
- "family insurance"
- "family members"
- "family policies"
- "family coverage"
- "my family"
- "our family"
- "family details"

### 2. Enhanced Name Recognition
The system can identify family members mentioned in queries using multiple name patterns:
- First Last (e.g., "Gloria Wood")
- First Middle Last (e.g., "John Michael Smith")
- First M. Last (e.g., "John M. Smith")
- First Last M. (e.g., "John Smith M.")

### 3. FamilyID Support
Users can also query by FamilyID directly (e.g., "Show me family FID-1027 insurance")

### 4. Comprehensive Family Information
When a family query is detected, the system retrieves:
- All family members' insurance policies
- Policy types and numbers
- Premium amounts
- Expiry dates
- Nominee information
- Family summary with total premiums

## Implementation Details

### Files Modified

1. **vector.py**
   - Added `is_family_query()` function for query classification
   - Added `get_family_members_by_name()` function
   - Added `get_family_members_by_family_id()` function
   - Added `get_family_summary()` function
   - Enhanced `enhanced_retriever()` function

2. **main_langraph_memory.py**
   - Updated RAG function to use enhanced retriever
   - Enhanced system prompts for better family insurance responses
   - Updated FT function with family insurance guidelines

### New Functions

#### `is_family_query(query)`
Detects if a query is related to family insurance.

#### `get_family_members_by_name(name, df)`
Retrieves all family members when given a person's name.

#### `get_family_members_by_family_id(family_id, df)`
Retrieves all family members by FamilyID.

#### `get_family_summary(family_id, df)`
Provides a comprehensive summary of family insurance information.

#### `enhanced_retriever(query)`
Main function that handles both family and regular insurance queries.

## Usage Examples

### Example Queries and Responses

1. **"What is my family insurance information?"**
   - System detects family query
   - Retrieves all family members' policies
   - Provides organized response by family member

2. **"Tell me about Gloria Wood's family insurance"**
   - System finds Gloria Wood
   - Retrieves all family members with same FamilyID (FID-1027)
   - Shows policies for Gloria Wood, Stacy Simpson, and Roger Morrison

3. **"Show me family FID-1027 insurance"**
   - Direct FamilyID query
   - Retrieves all members of that family
   - Provides comprehensive family summary

## Data Structure

The system uses the FamilyID field from the insurance data to group family members:

```csv
PolicyID,Name,DOB,PolicyNumber,InsuranceType,IssueDate,ExpiryDate,PremiumAmount,AccountNumber,IFSCCode,GSTNumber,FamilyID,NomineeName,Details
```

Example Family Group (FID-1027):
- Gloria Wood (Health Insurance)
- Stacy Simpson (Health & Travel Insurance)
- Roger Morrison (Health & Home Insurance)

## Response Format

When responding to family insurance queries, the system provides:

1. **Family Summary**
   - Total number of family members
   - Total premium amount
   - Types of insurance coverage

2. **Individual Member Details**
   - Name and policy information
   - Policy type and number
   - Premium amount and expiry date
   - Nominee information

3. **Organized Presentation**
   - Clear separation by family member
   - Consistent formatting
   - FamilyID reference for clarity

## Testing

A test script (`test_family_retrieval.py`) has been created to verify the functionality:

```bash
python test_family_retrieval.py
```

The test script validates:
- Family query detection
- Name recognition
- Family member retrieval
- Response formatting

## Benefits

1. **Comprehensive Information**: Users get complete family insurance overview
2. **Better Organization**: Information is clearly organized by family member
3. **Flexible Queries**: Multiple ways to ask about family insurance
4. **Consistent Responses**: Standardized format for family information
5. **Enhanced User Experience**: More helpful and informative responses

## Future Enhancements

Potential improvements could include:
- Family premium analysis and trends
- Policy expiry alerts for family members
- Family insurance recommendations
- Cross-family policy comparisons
- Family insurance portfolio optimization suggestions 