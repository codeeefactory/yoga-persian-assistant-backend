#!/usr/bin/env python3
"""
Backend service to add level headers to PDF files
This integrates with the main backend to automatically add level numbers to PDFs
"""

import fitz  # PyMuPDF
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

class PDFLevelHeaderAdder:
    """Class to add level headers to PDF files"""
    
    def __init__(self):
        self.header_height = 60  # Height of header in points
        self.font_size = 24
        self.header_bg_color = (0.95, 0.95, 0.95)  # Light gray
        self.border_color = (0.7, 0.7, 0.7)  # Darker gray
        self.text_color = (0, 0, 0)  # Black
    
    def add_level_header(self, pdf_path: str, level_number: str, output_path: Optional[str] = None) -> bool:
        """
        Add level header to PDF file
        
        Args:
            pdf_path: Path to input PDF
            level_number: Level number to add
            output_path: Output path (if None, overwrites original)
        
        Returns:
            bool: Success status
        """
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                self._add_header_to_page(page, level_number)
            
            # Save PDF
            if output_path is None:
                output_path = pdf_path
            
            doc.save(output_path)
            doc.close()
            
            return True
            
        except Exception as e:
            print(f"Error adding header to {pdf_path}: {e}")
            return False
    
    def _add_header_to_page(self, page, level_number: str):
        """Add header to a single page"""
        # Get page dimensions
        rect = page.rect
        width = rect.width
        height = rect.height
        
        # Create header rectangle
        header_rect = fitz.Rect(0, height - self.header_height, width, height)
        
        # Add background
        page.draw_rect(header_rect, color=self.header_bg_color, fill=self.header_bg_color)
        
        # Add border line
        page.draw_line(
            fitz.Point(0, height - self.header_height),
            fitz.Point(width, height - self.header_height),
            color=self.border_color,
            width=2
        )
        
        # Add level text (Persian)
        persian_text = f"سطح {level_number}"
        page.insert_text(
            fitz.Point(20, height - 25),
            persian_text,
            fontsize=self.font_size,
            color=self.text_color,
            fontname="helv"
        )
        
        # Add level text (English)
        english_text = f"Level {level_number}"
        page.insert_text(
            fitz.Point(width - 150, height - 25),
            english_text,
            fontsize=self.font_size - 4,
            color=self.text_color,
            fontname="helv"
        )
        
        # Add decorative elements
        self._add_decorative_elements(page, width, height)
    
    def _add_decorative_elements(self, page, width: float, height: float):
        """Add decorative elements to the header"""
        # Add small decorative lines
        y_pos = height - self.header_height + 10
        
        # Left decorative line
        page.draw_line(
            fitz.Point(10, y_pos),
            fitz.Point(30, y_pos),
            color=self.border_color,
            width=1
        )
        
        # Right decorative line
        page.draw_line(
            fitz.Point(width - 30, y_pos),
            fitz.Point(width - 10, y_pos),
            color=self.border_color,
            width=1
        )
    
    def process_pdf_with_level_detection(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process PDF by detecting level and adding header
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            dict: Processing result with level information
        """
        try:
            # Extract text to detect level
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Detect level number
            level_number = self._detect_level_from_text(text)
            
            # Create output path
            output_path = self._create_output_path(pdf_path, level_number)
            
            # Add header
            success = self.add_level_header(pdf_path, level_number, output_path)
            
            return {
                "success": success,
                "level_number": level_number,
                "original_path": pdf_path,
                "output_path": output_path,
                "detected_from_text": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original_path": pdf_path
            }
    
    def _detect_level_from_text(self, text: str) -> str:
        """Detect level number from PDF text"""
        import re
        
        # Level detection patterns
        patterns = [
            r'سطح\s*([۰-۹]+)',
            r'مرحله\s*([۰-۹]+)',
            r'بخش\s*([۰-۹]+)',
            r'سطح\s*(\d+)',
            r'مرحله\s*(\d+)',
            r'بخش\s*(\d+)',
            r'level\s*(\d+)',
            r'stage\s*(\d+)',
            r'section\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                level_num = match.group(1)
                
                # Convert Persian digits
                persian_digits = {
                    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
                    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
                }
                
                if level_num in persian_digits:
                    return persian_digits[level_num]
                else:
                    return level_num
        
        # Default to level 1 if no level detected
        return "1"
    
    def _create_output_path(self, original_path: str, level_number: str) -> str:
        """Create output path for modified PDF"""
        path = Path(original_path)
        return str(path.parent / f"{path.stem}_level_{level_number}{path.suffix}")

def add_level_headers_to_pdfs(pdf_directory: str) -> Dict[str, Any]:
    """
    Add level headers to all PDFs in a directory
    
    Args:
        pdf_directory: Directory containing PDF files
    
    Returns:
        dict: Processing results
    """
    pdf_dir = Path(pdf_directory)
    
    if not pdf_dir.exists():
        return {"success": False, "error": f"Directory {pdf_directory} not found"}
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        return {"success": False, "error": "No PDF files found"}
    
    adder = PDFLevelHeaderAdder()
    results = []
    
    for pdf_file in pdf_files:
        result = adder.process_pdf_with_level_detection(str(pdf_file))
        results.append(result)
    
    successful = sum(1 for r in results if r.get("success", False))
    
    return {
        "success": True,
        "total_files": len(pdf_files),
        "successful": successful,
        "failed": len(pdf_files) - successful,
        "results": results
    }

if __name__ == "__main__":
    # Test the functionality
    pdf_directory = "../samples/pdfs"
    result = add_level_headers_to_pdfs(pdf_directory)
    
    print("PDF Level Header Processing Results:")
    print(f"Total files: {result.get('total_files', 0)}")
    print(f"Successful: {result.get('successful', 0)}")
    print(f"Failed: {result.get('failed', 0)}")
    
    for res in result.get('results', []):
        if res.get('success'):
            print(f"✅ {res['original_path']} → Level {res['level_number']}")
        else:
            print(f"❌ {res['original_path']} → Error: {res.get('error', 'Unknown')}")
