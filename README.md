# ReimburSync 🚀

**AI-powered Invoice Reimbursement Checker using LLMs**

---

## 📌 Features

- 📄 Upload HR Reimbursement Policy (PDF)
- 📁 Upload ZIP file containing multiple employee invoice PDFs
- 🤖 Backend logic powered by Large Language Models (LLMs)
  - Extracts and analyzes invoice and policy content
  - Determines reimbursement status: Fully Reimbursed, Partially Reimbursed, or Declined
  - Calculates reimbursable amount
  - Provides detailed reasoning backed by policy clauses
- 📊 JSON output for each invoice with:
  - Invoice name
  - Status
  - Amount
  - Justification
- 📉 Optimized LLM prompt for minimal calls and maximum accuracy
- 🔐 Secure file handling and efficient document parsing

---

## 🛠 Technology Used

- **Frontend:** React.js  
- **Backend:** FastAPI (Python)  
- **LLM Integration:** Open-source LLM / Groq / Google AI Studio  
- **PDF & ZIP Handling:** PyMuPDF, zipfile, pdfminer  
- **Prompt Engineering:** Custom-crafted system prompt for strict policy adherence

---

## 👨‍💻 Developed by

**Shubham Murtadak**  
*AI Engineer*

---
