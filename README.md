# Yoga PDF Extractor Backend

A powerful Python FastAPI backend for extracting text and images from PDF files, specifically designed for yoga exercise content analysis.

## Features

- **Multiple PDF Processing Methods**:
  - PyMuPDF (fitz) - Fast text and image extraction
  - PDFMiner - Advanced text extraction with layout analysis
  - Tesseract OCR - Optical Character Recognition for scanned PDFs

- **Yoga Content Analysis**:
  - Automatic detection of yoga exercise names
  - Persian and English text recognition
  - Structured exercise data extraction

- **Image Extraction**:
  - High-quality image extraction from PDFs
  - Base64 encoding for web compatibility
  - Multiple image format support

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (optional, for OCR functionality)

### Quick Setup

1. **Run the setup script**:
   ```bash
   python setup.py
   ```

2. **Or install manually**:
   ```bash
   pip install -r requirements.txt
   ```

### Tesseract OCR Installation

- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Ubuntu**: `sudo apt-get install tesseract-ocr`

## Usage

### Start the Server

```bash
python run.py
```

The server will start at `http://localhost:8000`

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

### Health Check
```
GET /health
```

### Extract PDF Content
```
POST /extract/pdf
```
- **Parameters**:
  - `file`: PDF file upload
  - `method`: Extraction method (`pymupdf`, `pdfminer`, `ocr`, `all`)
  - `analyze_yoga`: Boolean to analyze for yoga content

### Extract Yoga Exercises
```
POST /extract/yoga-exercises
```
- **Parameters**:
  - `file`: PDF file upload
- **Returns**: Structured yoga exercise data

### Extract Images Only
```
POST /extract/images
```
- **Parameters**:
  - `file`: PDF file upload
- **Returns**: Base64 encoded images

## Example Usage

### Python Client
```python
import requests

# Upload and extract yoga exercises
with open('yoga_manual.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/extract/yoga-exercises', files=files)
    result = response.json()
    print(f"Found {result['totalExercises']} exercises")
```

### JavaScript Client
```javascript
const formData = new FormData();
formData.append('file', pdfFile);

const response = await fetch('http://localhost:8000/extract/yoga-exercises', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(`Found ${result.totalExercises} exercises`);
```

## Response Format

### Yoga Exercises Response
```json
{
  "success": true,
  "title": "تمرینات یوگا - استخراج شده",
  "subtitle": "PDF Extraction with Python",
  "contentType": "python_extracted",
  "sourceFile": "yoga_manual.pdf",
  "totalExercises": 15,
  "totalImages": 8,
  "extractionMethod": "Python FastAPI with PyMuPDF",
  "levels": [{
    "id": "python_extracted_level",
    "persianName": "تمرینات استخراج شده",
    "englishName": "Extracted Exercises",
    "exercises": [
      {
        "id": "yoga_exercise_1",
        "persianName": "حالت درخت",
        "englishName": "Tree Pose",
        "category": "Yoga Asana",
        "difficulty": "intermediate",
        "confidence": 0.8
      }
    ]
  }],
  "images": [
    {
      "id": "image_0_0",
      "dataUrl": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
      "width": 200,
      "height": 300,
      "page": 1
    }
  ]
}
```

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **PyMuPDF**: Fast PDF processing library
- **PDFMiner**: Advanced PDF text extraction
- **Tesseract**: OCR engine for text recognition
- **Pillow**: Python Imaging Library
- **Uvicorn**: ASGI server for FastAPI

## Configuration

The server runs on `localhost:8000` by default. You can modify the host and port in `run.py`:

```python
uvicorn.run(
    "main:app",
    host="0.0.0.0",  # Change to your desired host
    port=8000,       # Change to your desired port
    reload=True
)
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Install Tesseract OCR and ensure it's in your PATH
2. **Port already in use**: Change the port in `run.py` or kill the process using port 8000
3. **Memory issues with large PDFs**: The server processes PDFs in memory, very large files may cause issues

### Logs

The server logs all operations. Check the console output for detailed error messages.

## Development

### Adding New Extraction Methods

1. Add the method to the `PDFExtractor` class
2. Update the API endpoint to support the new method
3. Add tests for the new functionality

### Customizing Yoga Analysis

Modify the `analyze_yoga_content` method in `main.py` to improve exercise detection:

```python
def analyze_yoga_content(self, text: str) -> List[Dict[str, Any]]:
    # Add your custom logic here
    pass
```

## License

This project is part of the Yoga Persian Assistant application.
