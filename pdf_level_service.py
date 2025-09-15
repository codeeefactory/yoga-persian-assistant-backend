#!/usr/bin/env python3
"""
PDF Level Header Service
This service provides endpoints to add level headers to PDF files
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import tempfile
import os
from pathlib import Path
from typing import Optional
import json

# Import the PDF level header adder
from add_pdf_level_headers import PDFLevelHeaderAdder, add_level_headers_to_pdfs

app = FastAPI(title="PDF Level Header Service", version="1.0.0")

@app.post("/add-level-header")
async def add_level_header_to_pdf(
    file: UploadFile = File(...),
    level_number: Optional[str] = None
):
    """
    Add level header to a PDF file
    
    Args:
        file: PDF file to process
        level_number: Level number to add (if not provided, will be detected)
    
    Returns:
        Modified PDF file with level header
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        content = await file.read()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_input:
            temp_input.write(content)
            temp_input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # Initialize PDF level header adder
            adder = PDFLevelHeaderAdder()
            
            # Detect level if not provided
            if level_number is None:
                # Extract text to detect level
                import fitz
                doc = fitz.open(temp_input_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                
                level_number = adder._detect_level_from_text(text)
            
            # Add level header
            success = adder.add_level_header(temp_input_path, level_number, temp_output_path)
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to add level header")
            
            # Return the modified PDF
            return FileResponse(
                temp_output_path,
                media_type='application/pdf',
                filename=f"level_{level_number}_{file.filename}",
                headers={"Content-Disposition": f"attachment; filename=level_{level_number}_{file.filename}"}
            )
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
            except:
                pass
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/detect-level")
async def detect_level_from_pdf(file: UploadFile = File(...)):
    """
    Detect level number from PDF content
    
    Args:
        file: PDF file to analyze
    
    Returns:
        Detected level information
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract text and detect level
            import fitz
            doc = fitz.open(temp_file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Detect level
            adder = PDFLevelHeaderAdder()
            level_number = adder._detect_level_from_text(text)
            
            return {
                "success": True,
                "filename": file.filename,
                "detected_level": level_number,
                "persian_text": f"سطح {level_number}",
                "english_text": f"Level {level_number}",
                "text_sample": text[:500] + "..." if len(text) > 500 else text
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing PDF: {str(e)}")

@app.post("/process-directory")
async def process_pdf_directory(directory_path: str):
    """
    Process all PDFs in a directory and add level headers
    
    Args:
        directory_path: Path to directory containing PDF files
    
    Returns:
        Processing results
    """
    try:
        if not os.path.exists(directory_path):
            raise HTTPException(status_code=404, detail=f"Directory {directory_path} not found")
        
        # Process all PDFs in directory
        result = add_level_headers_to_pdfs(directory_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing directory: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "PDF Level Header Service",
        "version": "1.0.0",
        "endpoints": [
            "/add-level-header - Add level header to PDF",
            "/detect-level - Detect level from PDF content",
            "/process-directory - Process all PDFs in directory"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
