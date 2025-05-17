import os 
import zipfile
import tempfile
from llama_parse import LlamaParse
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()


GROQ_API_KEY=os.getenv("GROQ_API_KEY")
LLAMA_CLOUD_API_KEY=os.getenv("LLAMA_CLOUD_API_KEY")


#create instance of llamaparse
parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY
)

def parsed_pdf_data(pdf_path):
    print("Inside parsed pdf data fucntion !!!")
   
    instruction = """
    You will receive one of two document types: an HR Reimbursement Policy or an Employee Invoice. Your goal is to extract information accordingly.
    1. Detect Document Type:
    - If the document contains terms like "Policy", "Reimbursement Categories", or "Submission Process", treat it as an HR Policy.
    - If it contains terms like "Invoice ID", "Amount", "Tax", or "Subtotal", treat it as an Invoice.

    2. If Document is HR Policy:
    - Extract all text content in a continuous, readable format.
    - Preserve document structure:
    - Headings (e.g., Purpose, Scope, Food and Beverages)
    - Bullet points and numbered lists
    - Paragraph breaks
    - Do not break the content into JSON. Just return clean, structured text preserving section hierarchy.

    3. If Document is an Invoice:
    - Extract and return the following fields in JSON format and always use {{}} for all curly braces dont use {}:
    - Invoice ID
    - Invoice Date
    - Customer/Employee Name
    - Mobile Number (if any)
    - Service or Pickup Address (if any)
    - Items: list of charges if present
    - Taxes if applied
    - Total: final amount paid

    - If itemized charges are listed, include them under "Items" as description and amount.
    - If not itemized, leave "Items" as an empty list and include only "Total".
    - All monetary values must be numbers (e.g., ₹233 → 233.0)
    - Use {{ for opening curly brace and }} for closing curly brace 

    Example Output (Itemized):
    {{
    "DocumentType":"",
    "InvoiceID": "",
    "InvoiceDate": "",
    "CustomerName": "",
    "MobileNumber": "",
    "ServiceAddress": "",
    "Items": [
        {{"Description": "", "Amount": 0.0}},
        {{"Description": "", "Amount": 0.0}},
        
    ],
    "Total": 0.0
    }}

    Example Output (Non-itemized):
    {{
    "InvoiceID": "",
    "InvoiceDate": "",
    "CustomerName": "",
    "MobileNumber": "",
    "ServiceAddress": "",
    "Items": [],
    "Total": 0.0
    }}

    4. General Notes:
    - Ensure monetary values are extracted as numbers (e.g., ₹233 → 233.0)
    - Ignore any non-relevant decorative or boilerplate content
    - Maintain clarity and completeness across multi-page documents
    """


    try:
        # Use LlamaParse to extract the data
        parsed_data = LlamaParse(result_type="markdown", api_key=LLAMA_CLOUD_API_KEY,
                                parsing_instruction=instruction).load_data(pdf_path)


    except Exception as e:
        parsed_data=""
   
    return parsed_data



def evaluate(policy_data,invoices_data):
    # prompt_t="""
    # You are an expert HR manager whose job is to analyse all the invoices carefully that the user has sent and then you have to generate a reimbursement amount based on the company policy.
    # The invoices that you will get are approved from higher management. You just have to analyze and go through the text and see if it comes under the policy and does not do anything restricted.
    # Your job is to
    # - understand the invoice sent by user by carefully analyzing the invoice data
    # - break down data for better understanding
    # - Analyze the extracted policy text of the company's policy
    # - Understand what can be reimbursed and how much reimbursement is allowed with analyzing all the restrictions alongside
    # - The invoices is already Approved so just focus on
    # PROCESS (internal use):
    # 1. Carefully examine the invoice details (date, amount, purpose, items, taxes)
    # 2. Identify the appropriate expense category based on company policy
    # 3. Determine the applicable policy limit for this category
    # 4. Evaluate if any policy restrictions apply to this expense
    # 5. Calculate the reimbursable amount based on policy rules
    # DECISION CRITERIA:
    # - APPROVED: Invoice amount is within policy limits and meets all criteria
    # - PARTIAL: Invoice exceeds policy limits but is otherwise eligible (reimburse up to limit)
    # - DECLINED: Expense violates policy restrictions or is explicitly non-reimbursable
    
    # "Input Data**:
    # Policy Data :{policy_data}
    # Inovoice Data :{invoices_data}

    
    # **Strict Instruction**:Provide output only in below mentiond json output format do not provide anhthing else explanation ,process etc...
    # Output Format :
    # {{
    #  "Invoice identifier":"(e.g., filename)" ,
	#  "Reimbursement Status":" (Fully Reimbursed, Partially Reimbursed, Declined)."
	#  "Reimbursable Amount":" (integer value)."
	#  "Reason": "A clear explanation derived from the policy for why an invoice was Partially Reimbursed or Declined. Reasons should also be provided for Fully Reimbursed indicating which policy clause supports it."

    # }}
  
    # """

    prompt_t = """
    You are an expert HR manager responsible for analyzing approved invoices to determine the eligible reimbursement amount based strictly on the provided company policy.
    Each invoice you receive is pre-approved by higher management. Your task is to:
    - Carefully understand and break down the invoice data.
    - Analyze the provided company policy thoroughly.
    - Determine what can be reimbursed, how much, and ensure no policy violations.
    - Focus only on policy validation and reimbursement calculation.

    # PROCESS:
    1. Examine invoice details (date, amount, purpose, items, taxes).
    2. Identify the appropriate expense category as per policy.
    3. Check applicable policy limits for that category.
    4. Evaluate if any restrictions apply.
    5. Calculate the final reimbursable amount.

    ## DECISION CRITERIA:
    - APPROVED: Within policy limits and meets all criteria.
    - PARTIAL: Exceeds limits but is otherwise valid (reimburse up to limit).
    - DECLINED: Violates policy restrictions or is not reimbursable.

    **Strict Instruction**: Provide output ONLY in the JSON format below — no explanations, processes, or extra commentary.

    ## Output Format:
    {
    "Invoice identifier": "(e.g., filename)",
    "Reimbursement Status": "(Fully Reimbursed, Partially Reimbursed, Declined)",
    "Reimbursable Amount": (integer),
    "Reason": "Brief justification referring to relevant policy rules or limits."
    }

    # Input Data :
    Policy Data:
    {policy_data}

    Invoice Data:
    {invoices_data}
    """


    prompt=PromptTemplate(
            template=prompt_t,
            input_variables=['policy_data','invoices_data']
        )

    chain=prompt |llm

    # report=chain.invoke(dummy_data)
    report=chain.invoke({'policy_data':policy_data,'invoices_data':invoices_data})

    print("main report generated is :",report)
    report=report.content

    return report 


def process_documents(policy_path, zip_path):
    print("policy path is:", policy_path)
    print("zip path is:", zip_path)

    # Extract HR policy content (multi-doc structure)
    policy_docs = parsed_pdf_data(policy_path)
    print("extracted policy docs:", policy_docs)

    # Combine all text from HR policy docs
    if isinstance(policy_docs, list):
        policy_text = "".join([doc.text for doc in policy_docs])
    else:
        policy_text = policy_docs[0].text  # fallback

    # Extract invoices from zip into the same "temp" directory
    invoice_dir = os.path.join(os.path.dirname(zip_path), "invoices")
    os.makedirs(invoice_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(invoice_dir)

    invoice_results = []

    # Walk through all subdirectories and files
    for root, _, files in os.walk(invoice_dir):
        for fname in files:
            fpath = os.path.join(root, fname)

            # Skip unsupported file types
            if not fname.lower().endswith((".pdf",)):
                continue

            try:
                invoice_doc = parsed_pdf_data(fpath)
                invoice_text = invoice_doc[0].text

                invoice_results.append({
                    "invoice_filename": fname,
                    "parsed_text": invoice_text
                })
            except Exception as e:
                print(f"Error while parsing the file '{fpath}': {e}")


    # return policy_text,invoice_results
    result=evaluate(policy_text,invoice_results)

    return result






if __name__=='__main__':
    # policy_path="D:\\personal\\Interview-Assignments\\iae_soulutions\\Store\\task 1 dataset\\Policy-Nov-2024.pdf.docx"
    # policy_path="D:\\personal\\Interview-Assignments\\iae_soulutions\\Store\\task 1 dataset\\Cab Bills\\Book-cab-02.pdf"
    policy_path="D:\\personal\\Interview-Assignments\\iae_soulutions\\Store\\task 1 dataset\\Meal Invoice\\Meal Invoice 2.pdf"
    extracted_data=parsed_pdf_data(policy_path)
    data=extracted_data[0].text

    # data=""
    # for doc in extracted_data:
    #     data=data+doc.text
    


    print("extracted data is:",data)