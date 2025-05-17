import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from Source.analysis import process_documents


app = FastAPI()

@app.post("/upload/")
async def upload_files(policy_file: UploadFile = File(...),invoice_zip: UploadFile = File(...)):
    # Create temporary directory
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)


    try:
        # Save uploaded files
        policy_path = os.path.join(temp_dir, policy_file.filename)
        with open(policy_path, "wb") as f:
            f.write(await policy_file.read())

        zip_path = os.path.join(temp_dir, invoice_zip.filename)
        with open(zip_path, "wb") as f:
            f.write(await invoice_zip.read())

        print("policy path is :",policy_path)
        print("zip path is :",zip_path)

        # Call processing logic
        result = process_documents(policy_path, zip_path)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        shutil.rmtree(temp_dir)
