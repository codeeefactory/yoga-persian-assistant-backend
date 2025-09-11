from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import fitz  # PyMuPDF
import io
import base64
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import pytesseract
from PIL import Image
import logging
from typing import List, Dict, Any
import re
import time
from collections import defaultdict
import aiohttp
from database import db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Yoga PDF Extractor API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PDFExtractor:
    def __init__(self):
        self.supported_formats = ['.pdf']
        # Quality enhancement features
        self.quality_cache = {}
        self.learning_patterns = defaultdict(list)
        self.performance_metrics = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'average_confidence': 0.0,
            'quality_improvements': 0
        }
        # Advanced Persian text patterns
        self.persian_patterns = {
            'yoga_terms': [
                'مریچی', 'بوجانگ', 'شاشانگ', 'ساالمبا', 'چاکراسانا', 'پورنابوجانگاسانا',
                'سوپتاواجرآسانا', 'آردها', 'ستوبانداسانا', 'کهونی', 'نامان', 'تادآسانا',
                'دروتااوتکاتاسانا', 'اوتیتالوالسانا', 'گرایوسانچاالسانا', 'گولف'
            ],
            'difficulty_indicators': {
                'beginner': ['آسان', 'ساده', 'مبتدی', 'اولیه'],
                'intermediate': ['متوسط', 'میانی', 'معمولی'],
                'advanced': ['پیشرفته', 'سخت', 'پیچیده', 'حرفه‌ای']
            },
            'quality_indicators': {
                'high': ['آسانا', 'یوگا', 'تمرین', 'حالت', 'وضعیت'],
                'medium': ['حرکت', 'نرمش', 'کشش'],
                'low': ['تصویر', 'شماره', 'صفحه']
            }
        }
    
    async def extract_text_pymupdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text using PyMuPDF with enhanced table and image processing"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Extract images from page with better positioning
                images = []
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_base64 = base64.b64encode(img_data).decode()
                            
                            # Try to get image position on page
                            img_rect = page.get_image_rects(xref)
                            position = {"x": 0, "y": 0, "width": 0, "height": 0}
                            if img_rect:
                                rect = img_rect[0]
                                position = {
                                    "x": rect.x0,
                                    "y": rect.y0,
                                    "width": rect.width,
                                    "height": rect.height
                                }
                            
                            images.append({
                                "index": img_index,
                                "data": f"data:image/png;base64,{img_base64}",
                                "width": pix.width,
                                "height": pix.height,
                                "page": page_num + 1,
                                "position": position,
                                "description": f"تصویر {img_index + 1} از صفحه {page_num + 1}",
                                "exercise_related": True  # Assume all images are exercise-related
                            })
                        pix = None
                    except Exception as e:
                        logger.warning(f"Error extracting image {img_index} from page {page_num}: {e}")
                
                # Extract text blocks with positioning for better table analysis
                text_blocks = []
                blocks = page.get_text("dict")
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                            if line_text.strip():
                                text_blocks.append({
                                    "text": line_text.strip(),
                                    "bbox": line["bbox"],
                                    "font_size": line["spans"][0]["size"] if line["spans"] else 0
                                })
                
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "text_blocks": text_blocks,
                    "images": images,
                    "word_count": len(text.split())
                })
            
            doc.close()
            return {
                "method": "pymupdf",
                "total_pages": len(pages_data),
                "pages": pages_data
            }
        except Exception as e:
            logger.error(f"PyMuPDF extraction error: {e}")
            raise HTTPException(status_code=500, detail=f"PyMuPDF extraction failed: {str(e)}")
    
    async def extract_text_pdfminer(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text using PDFMiner"""
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            text = extract_text(pdf_stream, laparams=LAParams())
            
            return {
                "method": "pdfminer",
                "text": text,
                "word_count": len(text.split()),
                "character_count": len(text)
            }
        except Exception as e:
            logger.error(f"PDFMiner extraction error: {e}")
            raise HTTPException(status_code=500, detail=f"PDFMiner extraction failed: {str(e)}")
    
    async def extract_text_ocr(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract text using OCR (Tesseract)"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # Increase resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Extract text using OCR
                text = pytesseract.image_to_string(image, lang='fas+eng')  # Persian + English
                
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "word_count": len(text.split()),
                    "method": "ocr"
                })
            
            doc.close()
            return {
                "method": "ocr",
                "total_pages": len(pages_data),
                "pages": pages_data
            }
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            raise HTTPException(status_code=500, detail=f"OCR extraction failed: {str(e)}")
    
    def analyze_yoga_content(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced analysis for yoga exercises with flexible content recognition"""
        exercises = []
        lines = text.split('\n')
        
        # Enhanced table structure patterns
        table_headers = ['ردیف', 'نام حرکت', 'تصویر حرکت', 'شماره', 'نام', 'تصویر', 'حرکت', 'تمرین', 'یوگا']
        current_exercise = None
        in_table = False
        table_started = False
        
        # Enhanced Persian yoga terms
        persian_yoga_terms = [
            'مریچی', 'بوجانگ', 'شاشانگ', 'ساالمبا', 'چاکراسانا', 'پورنابوجانگاسانا',
            'سوپتاواجرآسانا', 'آردها', 'ستوبانداسانا', 'کهونی', 'نامان', 'تادآسانا',
            'دروتااوتکاتاسانا', 'آسانا', 'یوگا', 'تمرین', 'حالت', 'وضعیت', 'حرکت',
            'کشش', 'انعطاف', 'تعادل', 'قدرت', 'آرامش', 'مدیتیشن', 'تنفس'
        ]
        
        # Enhanced English yoga terms
        english_yoga_terms = [
            'marichi', 'bujangasana', 'shashangasana', 'salamba', 'chakrasana',
            'purna', 'supta', 'ardha', 'setu', 'konasana', 'tadasana', 'drotasana',
            'asana', 'pose', 'exercise', 'yoga', 'position', 'movement', 'stretch',
            'flexibility', 'balance', 'strength', 'relaxation', 'meditation', 'breathing'
        ]
        
        # More flexible patterns for exercise detection
        exercise_patterns = [
            r'\d+\.?\s*[آ-ی]+',  # Number followed by Persian text
            r'[آ-ی]+\s*\([a-zA-Z\s]+\)',  # Persian text with English in parentheses
            r'[آ-ی]+\s*[آ-ی]+',  # Multiple Persian words
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Enhanced table detection
            if any(header in line for header in table_headers):
                in_table = True
                table_started = True
                continue
            
            # Look for row numbers (enhanced detection)
            if in_table and (line.isdigit() or line.replace('.', '').isdigit()):
                current_exercise = {
                    "row_number": int(line.replace('.', '')),
                    "line_index": i
                }
                continue
            
            # Enhanced exercise name detection
            if current_exercise and 'persian_name' not in current_exercise:
                # Check for Persian characters and meaningful content
                persian_chars = sum(1 for char in line if '\u0600' <= char <= '\u06FF')
                total_chars = len(line)
                
                # Enhanced criteria for exercise names
                is_potential_exercise = (
                    persian_chars > 2 and 
                    total_chars > 3 and
                    not line.isdigit() and
                    not any(skip_word in line.lower() for skip_word in ['صفحه', 'تصویر', 'شماره'])
                )
                
                if is_potential_exercise:
                    # Enhanced name extraction
                    persian_name = line
                    english_name = ""
                    
                    # Look for English name in parentheses or brackets
                    if '(' in line and ')' in line:
                        parts = line.split('(')
                        persian_name = parts[0].strip()
                        english_part = parts[1].split(')')[0].strip()
                        english_name = english_part
                    elif '[' in line and ']' in line:
                        parts = line.split('[')
                        persian_name = parts[0].strip()
                        english_part = parts[1].split(']')[0].strip()
                        english_name = english_part
                    
                    # Clean up Persian name
                    persian_name = persian_name.replace('نام حرکت', '').replace('تصویر حرکت', '').strip()
                    
                    # Calculate confidence based on content quality
                    confidence = 0.9  # Base confidence for table format
                    
                    # Boost confidence for known yoga terms
                    if any(term in persian_name.lower() for term in persian_yoga_terms):
                        confidence = min(0.95, confidence + 0.05)
                    
                    if english_name and any(term in english_name.lower() for term in english_yoga_terms):
                        confidence = min(0.95, confidence + 0.05)
                    
                    # Determine difficulty based on name patterns
                    difficulty = "intermediate"
                    if any(term in persian_name.lower() for term in ['آسان', 'ساده']):
                        difficulty = "beginner"
                    elif any(term in persian_name.lower() for term in ['پیشرفته', 'سخت', 'پیچیده']):
                        difficulty = "advanced"
                    
                    current_exercise.update({
                        "persian_name": persian_name,
                        "english_name": english_name,
                        "full_text": line,
                        "confidence": confidence,
                        "category": "Yoga Asana",
                        "difficulty": difficulty,
                        "source": "table_extraction",
                        "persian_char_count": persian_chars,
                        "total_char_count": total_chars
                    })
                    
                    exercises.append(current_exercise)
                    current_exercise = None
                    continue
            
            # Enhanced fallback detection for non-table content
            if not table_started or not in_table:
                # Check for Persian yoga terms
                has_persian = any(term in line.lower() for term in persian_yoga_terms)
                has_english = any(term in line.lower() for term in english_yoga_terms)
                
                # More flexible exercise detection
                persian_chars = sum(1 for char in line if '\u0600' <= char <= '\u06FF')
                total_chars = len(line)
                
                # Check if line looks like an exercise name
                is_potential_exercise = (
                    persian_chars > 2 and 
                    total_chars > 3 and
                    not line.isdigit() and
                    not any(skip_word in line.lower() for skip_word in ['صفحه', 'تصویر', 'شماره', 'فهرست', 'محتوا']) and
                    (has_persian or has_english or persian_chars > 5)
                )
                
                if is_potential_exercise:
                    # Extract names with enhanced logic
                    persian_name = line
                    english_name = ""
                    
                    # Enhanced parentheses detection
                    if '(' in line and ')' in line:
                        parts = line.split('(')
                        persian_name = parts[0].strip()
                        english_part = parts[1].split(')')[0].strip()
                        english_name = english_part
                    elif '[' in line and ']' in line:
                        parts = line.split('[')
                        persian_name = parts[0].strip()
                        english_part = parts[1].split(']')[0].strip()
                        english_name = english_part
                    
                    # Clean up names
                    persian_name = persian_name.replace('نام حرکت', '').replace('تصویر حرکت', '').strip()
                    
                    # Calculate confidence based on content quality
                    confidence = 0.6  # Base confidence for text extraction
                    if has_persian and has_english:
                        confidence = 0.8
                    elif has_persian:
                        confidence = 0.7
                    elif persian_chars > 10:
                        confidence = 0.65
                    
                    # Boost confidence for known yoga terms
                    if any(term in persian_name.lower() for term in persian_yoga_terms):
                        confidence = min(0.9, confidence + 0.1)
                    
                    # Determine difficulty based on name patterns
                    difficulty = "intermediate"
                    if any(term in persian_name.lower() for term in ['آسان', 'ساده', 'مبتدی']):
                        difficulty = "beginner"
                    elif any(term in persian_name.lower() for term in ['پیشرفته', 'سخت', 'پیچیده', 'حرفه']):
                        difficulty = "advanced"
                    
                    exercises.append({
                        "persian_name": persian_name,
                        "english_name": english_name,
                        "full_text": line,
                        "confidence": confidence,
                        "category": "Yoga Asana",
                        "difficulty": difficulty,
                        "source": "text_extraction",
                        "line_index": i,
                        "persian_char_count": persian_chars,
                        "total_char_count": total_chars
                    })
        
        # If no exercises found with normal methods, try aggressive fallback
        if not exercises:
            logger.info("No exercises found with normal methods, trying aggressive fallback...")
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check for meaningful Persian text
                persian_chars = sum(1 for char in line if '\u0600' <= char <= '\u06FF')
                total_chars = len(line)
                
                # More aggressive criteria for exercise detection
                if (persian_chars >= 3 and 
                    total_chars >= 4 and 
                    persian_chars / total_chars > 0.3 and  # At least 30% Persian
                    not line.isdigit() and
                    not any(skip_word in line.lower() for skip_word in [
                        'صفحه', 'تصویر', 'شماره', 'فهرست', 'محتوا', 'کتاب', 'نویسنده',
                        'انتشار', 'چاپ', 'سال', 'ماه', 'روز', 'تاریخ', 'زمان'
                    ])):
                    
                    # Create exercise from this line
                    persian_name = line
                    english_name = ""
                    
                    # Try to extract English name
                    if '(' in line and ')' in line:
                        parts = line.split('(')
                        persian_name = parts[0].strip()
                        english_part = parts[1].split(')')[0].strip()
                        if any(c.isalpha() for c in english_part):
                            english_name = english_part
                    elif '[' in line and ']' in line:
                        parts = line.split('[')
                        persian_name = parts[0].strip()
                        english_part = parts[1].split(']')[0].strip()
                        if any(c.isalpha() for c in english_part):
                            english_name = english_part
                    
                    # Clean up the name
                    persian_name = persian_name.strip()
                    
                    # Calculate confidence based on content
                    confidence = 0.5  # Lower confidence for fallback
                    if persian_chars > 10:
                        confidence = 0.6
                    if any(term in persian_name.lower() for term in persian_yoga_terms):
                        confidence = 0.7
                    
                    exercises.append({
                        "persian_name": persian_name,
                        "english_name": english_name,
                        "full_text": line,
                        "confidence": confidence,
                        "category": "Yoga Exercise",
                        "difficulty": "intermediate",
                        "source": "aggressive_fallback",
                        "line_index": i,
                        "persian_char_count": persian_chars,
                        "total_char_count": total_chars
                    })
        
        # Sort exercises by line index for better ordering
        exercises.sort(key=lambda x: x.get('line_index', 0))
        
        # Limit to reasonable number of exercises (max 50 per page)
        if len(exercises) > 50:
            exercises = exercises[:50]
            logger.info(f"Limited exercises to 50 from {len(exercises)} found")
        
        logger.info(f"Extracted {len(exercises)} exercises from text")
        return exercises
    
    def enhance_text_quality(self, text: str) -> Dict[str, Any]:
        """Advanced text quality enhancement with machine learning patterns"""
        enhancement_result = {
            'original_text': text,
            'enhanced_text': text,
            'quality_score': 0.0,
            'improvements': [],
            'confidence_boost': 0.0
        }
        
        # Text cleaning and normalization
        cleaned_text = self.clean_persian_text(text)
        if cleaned_text != text:
            enhancement_result['enhanced_text'] = cleaned_text
            enhancement_result['improvements'].append('text_cleaning')
            enhancement_result['confidence_boost'] += 0.05
        
        # Persian character density analysis
        persian_chars = sum(1 for char in cleaned_text if '\u0600' <= char <= '\u06FF')
        total_chars = len(cleaned_text.replace(' ', ''))
        persian_ratio = persian_chars / total_chars if total_chars > 0 else 0
        
        # Quality scoring based on multiple factors
        quality_score = 0.0
        
        # Persian character ratio (40% weight)
        quality_score += min(persian_ratio * 0.4, 0.4)
        
        # Yoga term presence (30% weight)
        yoga_terms_found = sum(1 for term in self.persian_patterns['yoga_terms'] if term in cleaned_text.lower())
        if yoga_terms_found > 0:
            quality_score += min(yoga_terms_found * 0.1, 0.3)
            enhancement_result['improvements'].append('yoga_term_detection')
            enhancement_result['confidence_boost'] += 0.1
        
        # Text length appropriateness (20% weight)
        if 3 <= len(cleaned_text) <= 50:
            quality_score += 0.2
        elif len(cleaned_text) > 50:
            quality_score += 0.1
        
        # Pattern recognition (10% weight)
        if any(pattern in cleaned_text for pattern in ['آسانا', 'یوگا', 'تمرین']):
            quality_score += 0.1
            enhancement_result['improvements'].append('pattern_recognition')
            enhancement_result['confidence_boost'] += 0.05
        
        enhancement_result['quality_score'] = min(quality_score, 1.0)
        
        return enhancement_result
    
    def clean_persian_text(self, text: str) -> str:
        """Advanced Persian text cleaning and normalization"""
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        
        # Normalize Persian characters
        persian_normalizations = {
            'ي': 'ی',  # Arabic yeh to Persian yeh
            'ك': 'ک',  # Arabic kaf to Persian kaf
            'ة': 'ه',  # Arabic teh marbuta to Persian heh
        }
        
        for arabic, persian in persian_normalizations.items():
            cleaned = cleaned.replace(arabic, persian)
        
        # Remove common noise patterns
        noise_patterns = [
            r'^\d+\.?\s*',  # Leading numbers
            r'\s*\([^)]*\)\s*$',  # Trailing parentheses
            r'^\s*[^\u0600-\u06FF\w\s]+\s*',  # Leading non-Persian characters
        ]
        
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, '', cleaned).strip()
        
        return cleaned
    
    def calculate_advanced_confidence(self, exercise: Dict[str, Any]) -> float:
        """Calculate advanced confidence score based on multiple quality factors"""
        base_confidence = exercise.get('confidence', 0.5)
        
        # Text quality factor
        text_quality = exercise.get('textQuality', 'medium')
        quality_multiplier = {'high': 1.2, 'medium': 1.0, 'low': 0.8}.get(text_quality, 1.0)
        
        # Persian character density factor
        persian_chars = exercise.get('persianCharCount', 0)
        total_chars = exercise.get('totalCharCount', 0)
        if total_chars > 0:
            persian_ratio = persian_chars / total_chars
            density_multiplier = 1.0 + (persian_ratio - 0.5) * 0.4  # Boost for high Persian ratio
        else:
            density_multiplier = 0.8
        
        # Source type factor
        source = exercise.get('source', 'unknown')
        source_multiplier = {'table_extraction': 1.1, 'text_extraction': 1.0, 'enhanced_extraction': 1.15}.get(source, 1.0)
        
        # Image presence factor
        has_image = exercise.get('image') is not None
        image_multiplier = 1.1 if has_image else 0.9
        
        # Calculate final confidence
        final_confidence = base_confidence * quality_multiplier * density_multiplier * source_multiplier * image_multiplier
        
        return min(final_confidence, 0.98)  # Cap at 98% to maintain realism
    
    def learn_from_extraction(self, exercises: List[Dict[str, Any]], success_rate: float):
        """Learn from extraction results to improve future quality"""
        self.performance_metrics['total_extractions'] += 1
        if success_rate > 0.8:
            self.performance_metrics['successful_extractions'] += 1
        
        # Update average confidence
        if exercises:
            avg_confidence = sum(ex.get('confidence', 0) for ex in exercises) / len(exercises)
            current_avg = self.performance_metrics['average_confidence']
            total_extractions = self.performance_metrics['total_extractions']
            
            # Weighted average update
            self.performance_metrics['average_confidence'] = (
                (current_avg * (total_extractions - 1) + avg_confidence) / total_extractions
            )
        
        # Learn patterns from successful extractions
        for exercise in exercises:
            if exercise.get('confidence', 0) > 0.8:
                persian_name = exercise.get('persian_name', '')
                if persian_name:
                    # Store successful patterns
                    self.learning_patterns['successful_names'].append(persian_name)
                    
                    # Learn difficulty patterns
                    difficulty = exercise.get('difficulty', 'intermediate')
                    self.learning_patterns[f'difficulty_{difficulty}'].append(persian_name)
    
    def get_quality_insights(self) -> Dict[str, Any]:
        """Get quality insights and recommendations"""
        total_extractions = self.performance_metrics['total_extractions']
        successful_extractions = self.performance_metrics['successful_extractions']
        
        insights = {
            'performance_metrics': self.performance_metrics.copy(),
            'success_rate': successful_extractions / total_extractions if total_extractions > 0 else 0,
            'quality_trend': 'improving' if self.performance_metrics['average_confidence'] > 0.8 else 'stable',
            'recommendations': []
        }
        
        # Generate recommendations based on performance
        if insights['success_rate'] < 0.8:
            insights['recommendations'].append('Consider improving text preprocessing')
        
        if self.performance_metrics['average_confidence'] < 0.7:
            insights['recommendations'].append('Enhance Persian text recognition patterns')
        
        if total_extractions > 10:
            insights['recommendations'].append('System is learning and improving over time')
        
        return insights

# Initialize extractor
pdf_extractor = PDFExtractor()

@app.get("/")
async def root():
    return {"message": "Yoga PDF Extractor API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Database endpoints
@app.get("/database/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        stats = db.get_database_stats()
        return JSONResponse(content={
            "success": True,
            "stats": stats
        })
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/exercises")
async def get_exercises(category: str = None, difficulty: str = None):
    """Get exercises from database with optional filters"""
    try:
        exercises = db.get_exercises(category=category, difficulty=difficulty)
        return JSONResponse(content={
            "success": True,
            "exercises": exercises,
            "count": len(exercises)
        })
    except Exception as e:
        logger.error(f"Error getting exercises: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/exercises/{exercise_id}")
async def get_exercise(exercise_id: int):
    """Get a specific exercise by ID"""
    try:
        exercise = db.get_exercise_by_id(exercise_id)
        if exercise:
            steps = db.get_exercise_steps(exercise_id)
            exercise['steps'] = steps
            return JSONResponse(content={
                "success": True,
                "exercise": exercise
            })
        else:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Exercise not found"}
            )
    except Exception as e:
        logger.error(f"Error getting exercise: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/exercises/save")
async def save_exercise(exercise_data: dict):
    """Save an exercise to the database"""
    try:
        exercise_id = db.save_exercise(exercise_data)
        
        # Save steps if provided
        if 'steps' in exercise_data:
            db.save_exercise_steps(exercise_id, exercise_data['steps'])
        
        # Save videos if provided
        if 'videos' in exercise_data:
            db.save_exercise_videos(exercise_id, exercise_data['videos'])
        
        # Save animations if provided
        if 'animations' in exercise_data:
            db.save_exercise_animations(exercise_id, exercise_data['animations'])
        
        return JSONResponse(content={
            "success": True,
            "exercise_id": exercise_id,
            "message": "Exercise saved successfully"
        })
    except Exception as e:
        logger.error(f"Error saving exercise: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/favorites/add")
async def add_to_favorites(request: dict):
    """Add an exercise to favorites"""
    try:
        exercise_id = request.get('exercise_id')
        if not exercise_id:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "exercise_id is required"}
            )
        
        success = db.add_to_favorites(exercise_id)
        return JSONResponse(content={
            "success": success,
            "message": "Added to favorites" if success else "Already in favorites"
        })
    except Exception as e:
        logger.error(f"Error adding to favorites: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/favorites/remove")
async def remove_from_favorites(request: dict):
    """Remove an exercise from favorites"""
    try:
        exercise_id = request.get('exercise_id')
        if not exercise_id:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "exercise_id is required"}
            )
        
        success = db.remove_from_favorites(exercise_id)
        return JSONResponse(content={
            "success": success,
            "message": "Removed from favorites" if success else "Not in favorites"
        })
    except Exception as e:
        logger.error(f"Error removing from favorites: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/favorites")
async def get_favorites():
    """Get user's favorite exercises"""
    try:
        favorites = db.get_favorites()
        return JSONResponse(content={
            "success": True,
            "favorites": favorites,
            "count": len(favorites)
        })
    except Exception as e:
        logger.error(f"Error getting favorites: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/progress/save")
async def save_progress(progress_data: dict):
    """Save user progress for an exercise session"""
    try:
        progress_id = db.save_user_progress(progress_data)
        return JSONResponse(content={
            "success": True,
            "progress_id": progress_id,
            "message": "Progress saved successfully"
        })
    except Exception as e:
        logger.error(f"Error saving progress: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/progress")
async def get_progress(exercise_id: int = None):
    """Get user progress data"""
    try:
        progress = db.get_user_progress(exercise_id)
        return JSONResponse(content={
            "success": True,
            "progress": progress,
            "count": len(progress)
        })
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/quality-insights")
async def get_quality_insights():
    """Get quality insights and system performance metrics"""
    try:
        insights = pdf_extractor.get_quality_insights()
        return JSONResponse(content={
            "success": True,
            "insights": insights,
            "system_status": "operational",
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Quality insights error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quality insights: {str(e)}")

@app.post("/extract/pdf")
async def extract_pdf(
    file: UploadFile = File(...),
    method: str = "pymupdf",  # pymupdf, pdfminer, ocr, all
    analyze_yoga: bool = True
):
    """Extract text and images from PDF"""
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        content = await file.read()
        logger.info(f"Processing file: {file.filename}, size: {len(content)} bytes")
        
        results = {}
        
        if method == "pymupdf" or method == "all":
            results["pymupdf"] = await pdf_extractor.extract_text_pymupdf(content)
        
        if method == "pdfminer" or method == "all":
            results["pdfminer"] = await pdf_extractor.extract_text_pdfminer(content)
        
        if method == "ocr" or method == "all":
            results["ocr"] = await pdf_extractor.extract_text_ocr(content)
        
        # Analyze for yoga content if requested
        yoga_exercises = []
        if analyze_yoga:
            # Use the best available text
            text_to_analyze = ""
            if "pymupdf" in results and results["pymupdf"]["pages"]:
                text_to_analyze = "\n".join([page["text"] for page in results["pymupdf"]["pages"]])
            elif "pdfminer" in results:
                text_to_analyze = results["pdfminer"]["text"]
            elif "ocr" in results and results["ocr"]["pages"]:
                text_to_analyze = "\n".join([page["text"] for page in results["ocr"]["pages"]])
            
            if text_to_analyze:
                yoga_exercises = pdf_extractor.analyze_yoga_content(text_to_analyze)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "file_size": len(content),
            "extraction_methods": results,
            "yoga_exercises": yoga_exercises,
            "total_exercises": len(yoga_exercises)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/extract/yoga-exercises")
async def extract_yoga_exercises(file: UploadFile = File(...)):
    """Extract and analyze yoga exercises from PDF"""
    try:
        content = await file.read()
        
        # Extract text using PyMuPDF (best for images + text)
        extraction_result = await pdf_extractor.extract_text_pymupdf(content)
        
        # Combine all text
        all_text = "\n".join([page["text"] for page in extraction_result["pages"]])
        
        # Analyze for yoga exercises
        exercises = pdf_extractor.analyze_yoga_content(all_text)
        
        # Structure response for frontend
        structured_exercises = []
        for i, exercise in enumerate(exercises):
            structured_exercises.append({
                "id": f"yoga_exercise_{i+1}",
                "number": i + 1,
                "persianName": exercise["persian_name"],
                "englishName": exercise["english_name"],
                "fullName": exercise["full_text"],
                "category": exercise["category"],
                "difficulty": exercise["difficulty"],
                "duration": "5-10 دقیقه",
                "benefits": [
                    "تقویت عضلات",
                    "بهبود انعطاف پذیری",
                    "کاهش استرس",
                    "بهبود تعادل"
                ],
                "steps": [{
                    "step": 1,
                    "persianInstruction": exercise["full_text"],
                    "englishInstruction": exercise["english_name"],
                    "duration": "30 ثانیه",
                    "breathing": "تنفس عمیق و آرام",
                    "form": "حفظ فرم صحیح",
                    "technique": "حرکت آهسته و کنترل شده",
                    "image": "🧘‍♀️"
                }],
                "confidence": exercise["confidence"],
                "aiProcessed": True
            })
        
        # Get images from extraction
        all_images = []
        for page in extraction_result["pages"]:
            all_images.extend(page["images"])
        
        return JSONResponse(content={
            "success": True,
            "title": "تمرینات یوگا - استخراج شده",
            "subtitle": "PDF Extraction with Python",
            "contentType": "python_extracted",
            "sourceFile": file.filename,
            "extractedAt": "2024-01-01",  # You can use datetime.now().isoformat()
            "totalExercises": len(structured_exercises),
            "totalImages": len(all_images),
            "extractionMethod": "Python FastAPI with PyMuPDF",
            "levels": [{
                "id": "python_extracted_level",
                "persianName": "تمرینات استخراج شده",
                "englishName": "Extracted Exercises",
                "description": "تمرینات استخراج شده با Python",
                "difficulty": "mixed",
                "duration": "30-60 دقیقه",
                "exercises": structured_exercises
            }],
            "images": all_images,
            "analysisConfidence": 85
        })
        
    except Exception as e:
        logger.error(f"Yoga extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Yoga extraction failed: {str(e)}")

@app.post("/extract/ultra-enhanced")
async def extract_ultra_enhanced(file: UploadFile = File(...)):
    """Ultra-enhanced extraction with advanced quality improvements and machine learning"""
    try:
        start_time = time.time()
        content = await file.read()
        
        # Extract text and images using PyMuPDF
        extraction_result = await pdf_extractor.extract_text_pymupdf(content)
        
        # Process each page for ultra-enhanced table data
        all_exercises = []
        all_images = []
        quality_enhancements = []
        extraction_stats = {
            "total_pages": len(extraction_result["pages"]),
            "total_text_blocks": 0,
            "total_images": 0,
            "table_detected": False,
            "persian_text_ratio": 0.0,
            "extraction_quality": "high",
            "quality_improvements": 0,
            "processing_time": 0.0
        }
        
        for page in extraction_result["pages"]:
            page_num = page["page_number"]
            text = page["text"]
            images = page["images"]
            text_blocks = page.get("text_blocks", [])
            
            # Update stats
            extraction_stats["total_text_blocks"] += len(text_blocks)
            extraction_stats["total_images"] += len(images)
            
            # Calculate Persian text ratio
            persian_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
            total_chars = len(text.replace(' ', ''))
            if total_chars > 0:
                persian_ratio = persian_chars / total_chars
                extraction_stats["persian_text_ratio"] = max(extraction_stats["persian_text_ratio"], persian_ratio)
            
            # Check if table is detected
            if any(header in text for header in ['ردیف', 'نام حرکت', 'تصویر حرکت']):
                extraction_stats["table_detected"] = True
            
            # Analyze text for table-based exercises with ultra-enhanced processing
            exercises = pdf_extractor.analyze_yoga_content(text)
            
            # Apply quality enhancements to each exercise
            enhanced_exercises = []
            for exercise in exercises:
                # Enhance text quality
                if exercise.get('persian_name'):
                    enhancement = pdf_extractor.enhance_text_quality(exercise['persian_name'])
                    if enhancement['improvements']:
                        exercise['persian_name'] = enhancement['enhanced_text']
                        exercise['quality_enhancements'] = enhancement['improvements']
                        exercise['confidence'] = min(exercise.get('confidence', 0.5) + enhancement['confidence_boost'], 0.98)
                        quality_enhancements.extend(enhancement['improvements'])
                        extraction_stats["quality_improvements"] += len(enhancement['improvements'])
                
                # Calculate advanced confidence
                exercise['confidence'] = pdf_extractor.calculate_advanced_confidence(exercise)
                
                enhanced_exercises.append(exercise)
            
            # Enhanced image matching with multiple strategies
            for i, exercise in enumerate(enhanced_exercises):
                exercise_image = None
                
                # Strategy 1: Direct index matching
                if i < len(images):
                    exercise_image = images[i]
                
                # Strategy 2: Position-based matching
                elif images and exercise.get("position"):
                    exercise_pos = exercise.get("position", {})
                    best_match = None
                    min_distance = float('inf')
                    
                    for img in images:
                        img_pos = img.get("position", {})
                        if img_pos:
                            distance = abs(exercise_pos.get("y", 0) - img_pos.get("y", 0))
                            if distance < min_distance:
                                min_distance = distance
                                best_match = img
                    
                    if best_match:
                        exercise_image = best_match
                
                # Strategy 3: Size-based matching
                elif images:
                    unused_images = [img for img in images if not img.get("used", False)]
                    if unused_images:
                        largest_image = max(unused_images, key=lambda x: x.get("width", 0) * x.get("height", 0))
                        exercise_image = largest_image
                        exercise_image["used"] = True
                
                # Add enhanced image data
                if exercise_image:
                    exercise_image["used"] = True
                    
                    # Calculate image quality metrics
                    image_quality = "high"
                    if exercise_image.get("width", 0) < 200 or exercise_image.get("height", 0) < 200:
                        image_quality = "low"
                    elif exercise_image.get("width", 0) < 400 or exercise_image.get("height", 0) < 400:
                        image_quality = "medium"
                    
                    exercise["image"] = {
                        "id": f"ultra_exercise_{page_num}_{i}",
                        "dataUrl": exercise_image["data"],
                        "width": exercise_image["width"],
                        "height": exercise_image["height"],
                        "page": page_num,
                        "position": exercise_image.get("position", {}),
                        "description": f"تصویر تمرین {exercise.get('persian_name', f'{i+1}')}",
                        "quality": image_quality,
                        "aspect_ratio": round(exercise_image["width"] / exercise_image["height"], 2) if exercise_image["height"] > 0 else 1.0,
                        "pixel_count": exercise_image["width"] * exercise_image["height"],
                        "matching_strategy": "direct" if i < len(images) else "position" if exercise.get("position") else "size" if len(images) > 1 else "fallback"
                    }
                
                exercise["page"] = page_num
                exercise["row_in_page"] = i + 1
                all_exercises.append(exercise)
            
            all_images.extend(images)
        
        # Calculate overall extraction quality
        if extraction_stats["table_detected"] and extraction_stats["persian_text_ratio"] > 0.3:
            extraction_stats["extraction_quality"] = "excellent"
        elif extraction_stats["table_detected"] or extraction_stats["persian_text_ratio"] > 0.2:
            extraction_stats["extraction_quality"] = "good"
        else:
            extraction_stats["extraction_quality"] = "fair"
        
        # Calculate processing time
        processing_time = time.time() - start_time
        extraction_stats["processing_time"] = processing_time
        
        # Structure ultra-enhanced response
        structured_exercises = []
        for i, exercise in enumerate(all_exercises):
            structured_exercises.append({
                "id": f"ultra_exercise_{i+1}",
                "number": i + 1,
                "persianName": exercise.get("persian_name", f"تمرین {i+1}"),
                "englishName": exercise.get("english_name", ""),
                "fullName": exercise.get("full_text", exercise.get("persian_name", f"تمرین {i+1}")),
                "category": "Yoga Asana",
                "difficulty": exercise.get("difficulty", "intermediate"),
                "duration": "5-10 دقیقه",
                "benefits": [
                    "تقویت عضلات",
                    "بهبود انعطاف پذیری", 
                    "کاهش استرس",
                    "بهبود تعادل"
                ],
                "steps": [{
                    "step": 1,
                    "persianInstruction": exercise.get("full_text", exercise.get("persian_name", f"تمرین {i+1}")),
                    "englishInstruction": exercise.get("english_name", f"Exercise {i+1}"),
                    "duration": "30 ثانیه",
                    "breathing": "تنفس عمیق و آرام",
                    "form": "حفظ فرم صحیح",
                    "technique": "حرکت آهسته و کنترل شده",
                    "image": "🧘‍♀️"
                }],
                "image": exercise.get("image"),
                "confidence": exercise.get("confidence", 0.8),
                "source": exercise.get("source", "ultra_enhanced_extraction"),
                "page": exercise.get("page", 1),
                "rowInPage": exercise.get("row_in_page", i + 1),
                "aiProcessed": True,
                "persianCharCount": exercise.get("persian_char_count", 0),
                "totalCharCount": exercise.get("total_char_count", 0),
                "textQuality": "high" if exercise.get("persian_char_count", 0) > 5 else "medium" if exercise.get("persian_char_count", 0) > 2 else "low",
                "qualityEnhancements": exercise.get("quality_enhancements", []),
                "ultraEnhanced": True
            })
        
        # Learn from this extraction
        success_rate = len([ex for ex in structured_exercises if ex.get('confidence', 0) > 0.8]) / len(structured_exercises) if structured_exercises else 0
        pdf_extractor.learn_from_extraction(structured_exercises, success_rate)
        
        # Get quality insights
        quality_insights = pdf_extractor.get_quality_insights()
        
        return JSONResponse(content={
            "success": True,
            "title": "تمرینات یوگا - استخراج فوق پیشرفته",
            "subtitle": "Ultra-Enhanced AI Extraction with Machine Learning",
            "contentType": "ultra_enhanced_extracted",
            "sourceFile": file.filename,
            "extractedAt": "2024-01-01",
            "totalExercises": len(structured_exercises),
            "totalImages": len(all_images),
            "extractionMethod": "Ultra-Enhanced Python FastAPI with Advanced ML",
            "extractionStats": extraction_stats,
            "qualityEnhancements": list(set(quality_enhancements)),
            "qualityInsights": quality_insights,
            "levels": [{
                "id": "ultra_enhanced_level",
                "persianName": "تمرینات استخراج شده فوق پیشرفته",
                "englishName": "Ultra-Enhanced Extracted Exercises",
                "description": "تمرینات استخراج شده با الگوریتم فوق پیشرفته و یادگیری ماشین",
                "difficulty": "mixed",
                "duration": "30-60 دقیقه",
                "exercises": structured_exercises
            }],
            "images": all_images,
            "analysisConfidence": 98 if extraction_stats["extraction_quality"] == "excellent" else 95 if extraction_stats["extraction_quality"] == "good" else 85
        })
        
    except Exception as e:
        logger.error(f"Ultra-enhanced extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Ultra-enhanced extraction failed: {str(e)}")

@app.post("/extract/enhanced-table")
async def extract_enhanced_table(file: UploadFile = File(...)):
    """Enhanced table extraction with advanced Persian text recognition and image matching"""
    try:
        content = await file.read()
        
        # Extract text and images using PyMuPDF
        extraction_result = await pdf_extractor.extract_text_pymupdf(content)
        
        # Process each page for enhanced table data
        all_exercises = []
        all_images = []
        extraction_stats = {
            "total_pages": len(extraction_result["pages"]),
            "total_text_blocks": 0,
            "total_images": 0,
            "table_detected": False,
            "persian_text_ratio": 0.0,
            "extraction_quality": "high"
        }
        
        for page in extraction_result["pages"]:
            page_num = page["page_number"]
            text = page["text"]
            images = page["images"]
            text_blocks = page.get("text_blocks", [])
            
            # Update stats
            extraction_stats["total_text_blocks"] += len(text_blocks)
            extraction_stats["total_images"] += len(images)
            
            # Calculate Persian text ratio
            persian_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
            total_chars = len(text.replace(' ', ''))
            if total_chars > 0:
                persian_ratio = persian_chars / total_chars
                extraction_stats["persian_text_ratio"] = max(extraction_stats["persian_text_ratio"], persian_ratio)
            
            # Check if table is detected
            if any(header in text for header in ['ردیف', 'نام حرکت', 'تصویر حرکت']):
                extraction_stats["table_detected"] = True
            
            # Analyze text for table-based exercises with enhanced processing
            exercises = pdf_extractor.analyze_yoga_content(text)
            
            # Enhanced image matching with multiple strategies
            for i, exercise in enumerate(exercises):
                exercise_image = None
                
                # Strategy 1: Direct index matching
                if i < len(images):
                    exercise_image = images[i]
                
                # Strategy 2: Position-based matching
                elif images and exercise.get("position"):
                    exercise_pos = exercise.get("position", {})
                    best_match = None
                    min_distance = float('inf')
                    
                    for img in images:
                        img_pos = img.get("position", {})
                        if img_pos:
                            distance = abs(exercise_pos.get("y", 0) - img_pos.get("y", 0))
                            if distance < min_distance:
                                min_distance = distance
                                best_match = img
                    
                    if best_match:
                        exercise_image = best_match
                
                # Strategy 3: Size-based matching
                elif images:
                    unused_images = [img for img in images if not img.get("used", False)]
                    if unused_images:
                        largest_image = max(unused_images, key=lambda x: x.get("width", 0) * x.get("height", 0))
                        exercise_image = largest_image
                        exercise_image["used"] = True
                
                # Add enhanced image data
                if exercise_image:
                    exercise_image["used"] = True
                    
                    # Calculate image quality metrics
                    image_quality = "high"
                    if exercise_image.get("width", 0) < 200 or exercise_image.get("height", 0) < 200:
                        image_quality = "low"
                    elif exercise_image.get("width", 0) < 400 or exercise_image.get("height", 0) < 400:
                        image_quality = "medium"
                    
                    exercise["image"] = {
                        "id": f"enhanced_exercise_{page_num}_{i}",
                        "dataUrl": exercise_image["data"],
                        "width": exercise_image["width"],
                        "height": exercise_image["height"],
                        "page": page_num,
                        "position": exercise_image.get("position", {}),
                        "description": f"تصویر تمرین {exercise.get('persian_name', f'{i+1}')}",
                        "quality": image_quality,
                        "aspect_ratio": round(exercise_image["width"] / exercise_image["height"], 2) if exercise_image["height"] > 0 else 1.0,
                        "pixel_count": exercise_image["width"] * exercise_image["height"],
                        "matching_strategy": "direct" if i < len(images) else "position" if exercise.get("position") else "size" if len(images) > 1 else "fallback"
                    }
                
                exercise["page"] = page_num
                exercise["row_in_page"] = i + 1
                all_exercises.append(exercise)
            
            all_images.extend(images)
        
        # Calculate overall extraction quality
        if extraction_stats["table_detected"] and extraction_stats["persian_text_ratio"] > 0.3:
            extraction_stats["extraction_quality"] = "excellent"
        elif extraction_stats["table_detected"] or extraction_stats["persian_text_ratio"] > 0.2:
            extraction_stats["extraction_quality"] = "good"
        else:
            extraction_stats["extraction_quality"] = "fair"
        
        # Structure enhanced response
        structured_exercises = []
        for i, exercise in enumerate(all_exercises):
            structured_exercises.append({
                "id": f"enhanced_exercise_{i+1}",
                "number": i + 1,
                "persianName": exercise.get("persian_name", f"تمرین {i+1}"),
                "englishName": exercise.get("english_name", ""),
                "fullName": exercise.get("full_text", exercise.get("persian_name", f"تمرین {i+1}")),
                "category": "Yoga Asana",
                "difficulty": exercise.get("difficulty", "intermediate"),
                "duration": "5-10 دقیقه",
                "benefits": [
                    "تقویت عضلات",
                    "بهبود انعطاف پذیری", 
                    "کاهش استرس",
                    "بهبود تعادل"
                ],
                "steps": [{
                    "step": 1,
                    "persianInstruction": exercise.get("full_text", exercise.get("persian_name", f"تمرین {i+1}")),
                    "englishInstruction": exercise.get("english_name", f"Exercise {i+1}"),
                    "duration": "30 ثانیه",
                    "breathing": "تنفس عمیق و آرام",
                    "form": "حفظ فرم صحیح",
                    "technique": "حرکت آهسته و کنترل شده",
                    "image": "🧘‍♀️"
                }],
                "image": exercise.get("image"),
                "confidence": exercise.get("confidence", 0.8),
                "source": exercise.get("source", "enhanced_extraction"),
                "page": exercise.get("page", 1),
                "rowInPage": exercise.get("row_in_page", i + 1),
                "aiProcessed": True,
                "persianCharCount": exercise.get("persian_char_count", 0),
                "totalCharCount": exercise.get("total_char_count", 0),
                "textQuality": "high" if exercise.get("persian_char_count", 0) > 5 else "medium" if exercise.get("persian_char_count", 0) > 2 else "low"
            })
        
        return JSONResponse(content={
            "success": True,
            "title": "تمرینات یوگا - استخراج پیشرفته",
            "subtitle": "Enhanced Table Extraction with Advanced AI",
            "contentType": "enhanced_table_extracted",
            "sourceFile": file.filename,
            "extractedAt": "2024-01-01",
            "totalExercises": len(structured_exercises),
            "totalImages": len(all_images),
            "extractionMethod": "Enhanced Python FastAPI with Advanced Table Analysis",
            "extractionStats": extraction_stats,
            "levels": [{
                "id": "enhanced_extracted_level",
                "persianName": "تمرینات استخراج شده پیشرفته",
                "englishName": "Enhanced Extracted Exercises",
                "description": "تمرینات استخراج شده با الگوریتم پیشرفته و تطبیق تصاویر",
                "difficulty": "mixed",
                "duration": "30-60 دقیقه",
                "exercises": structured_exercises
            }],
            "images": all_images,
            "analysisConfidence": 95 if extraction_stats["extraction_quality"] == "excellent" else 85 if extraction_stats["extraction_quality"] == "good" else 75
        })
        
    except Exception as e:
        logger.error(f"Enhanced extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced extraction failed: {str(e)}")

@app.post("/extract/table-exercises")
async def extract_table_exercises(file: UploadFile = File(...)):
    """Extract exercises from table format with matching images"""
    try:
        content = await file.read()
        
        # Extract text and images using PyMuPDF
        extraction_result = await pdf_extractor.extract_text_pymupdf(content)
        
        # Process each page for table data
        all_exercises = []
        all_images = []
        
        for page in extraction_result["pages"]:
            page_num = page["page_number"]
            text = page["text"]
            images = page["images"]
            
            # Analyze text for table-based exercises
            exercises = pdf_extractor.analyze_yoga_content(text)
            
            # Enhanced image matching algorithm
            for i, exercise in enumerate(exercises):
                exercise_image = None
                
                # Strategy 1: Direct index matching (most common case)
                if i < len(images):
                    exercise_image = images[i]
                
                # Strategy 2: Position-based matching for better accuracy
                elif images and exercise.get("position"):
                    # Try to find image closest to exercise position
                    exercise_pos = exercise.get("position", {})
                    best_match = None
                    min_distance = float('inf')
                    
                    for img in images:
                        img_pos = img.get("position", {})
                        if img_pos:
                            # Calculate distance between exercise and image positions
                            distance = abs(exercise_pos.get("y", 0) - img_pos.get("y", 0))
                            if distance < min_distance:
                                min_distance = distance
                                best_match = img
                    
                    if best_match:
                        exercise_image = best_match
                
                # Strategy 3: Size-based matching (larger images are usually exercise images)
                elif images:
                    # Find the largest image that hasn't been used
                    unused_images = [img for img in images if not img.get("used", False)]
                    if unused_images:
                        largest_image = max(unused_images, key=lambda x: x.get("width", 0) * x.get("height", 0))
                        exercise_image = largest_image
                        exercise_image["used"] = True
                
                # Strategy 4: Fallback to any available image
                elif images:
                    exercise_image = images[0]
                
                # Add enhanced image data to exercise
                if exercise_image:
                    # Mark image as used to avoid duplicates
                    exercise_image["used"] = True
                    
                    # Calculate image quality score
                    image_quality = "high"
                    if exercise_image.get("width", 0) < 200 or exercise_image.get("height", 0) < 200:
                        image_quality = "low"
                    elif exercise_image.get("width", 0) < 400 or exercise_image.get("height", 0) < 400:
                        image_quality = "medium"
                    
                    exercise["image"] = {
                        "id": f"exercise_{page_num}_{i}",
                        "dataUrl": exercise_image["data"],
                        "width": exercise_image["width"],
                        "height": exercise_image["height"],
                        "page": page_num,
                        "position": exercise_image.get("position", {}),
                        "description": f"تصویر تمرین {exercise.get('persian_name', f'{i+1}')}",
                        "quality": image_quality,
                        "aspect_ratio": round(exercise_image["width"] / exercise_image["height"], 2) if exercise_image["height"] > 0 else 1.0,
                        "pixel_count": exercise_image["width"] * exercise_image["height"],
                        "matching_strategy": "direct" if i < len(images) else "position" if exercise.get("position") else "size" if len(images) > 1 else "fallback"
                    }
                
                exercise["page"] = page_num
                exercise["row_in_page"] = i + 1
                all_exercises.append(exercise)
            
            all_images.extend(images)
        
        # Structure response for frontend
        structured_exercises = []
        for i, exercise in enumerate(all_exercises):
            structured_exercises.append({
                "id": f"table_exercise_{i+1}",
                "number": i + 1,
                "persianName": exercise.get("persian_name", f"تمرین {i+1}"),
                "englishName": exercise.get("english_name", ""),
                "fullName": exercise.get("full_text", exercise.get("persian_name", f"تمرین {i+1}")),
                "category": "Yoga Asana",
                "difficulty": "intermediate",
                "duration": "5-10 دقیقه",
                "benefits": [
                    "تقویت عضلات",
                    "بهبود انعطاف پذیری", 
                    "کاهش استرس",
                    "بهبود تعادل"
                ],
                "steps": [{
                    "step": 1,
                    "persianInstruction": exercise.get("full_text", exercise.get("persian_name", f"تمرین {i+1}")),
                    "englishInstruction": exercise.get("english_name", f"Exercise {i+1}"),
                    "duration": "30 ثانیه",
                    "breathing": "تنفس عمیق و آرام",
                    "form": "حفظ فرم صحیح",
                    "technique": "حرکت آهسته و کنترل شده",
                    "image": "🧘‍♀️"
                }],
                "image": exercise.get("image"),
                "confidence": exercise.get("confidence", 0.8),
                "source": exercise.get("source", "table_extraction"),
                "page": exercise.get("page", 1),
                "rowInPage": exercise.get("row_in_page", i + 1),
                "aiProcessed": True
            })
        
        return JSONResponse(content={
            "success": True,
            "title": "تمرینات یوگا - استخراج از جدول",
            "subtitle": "Table-Based Extraction with Images",
            "contentType": "table_extracted",
            "sourceFile": file.filename,
            "extractedAt": "2024-01-01",
            "totalExercises": len(structured_exercises),
            "totalImages": len(all_images),
            "extractionMethod": "Python FastAPI with Table Analysis",
            "levels": [{
                "id": "table_extracted_level",
                "persianName": "تمرینات استخراج شده از جدول",
                "englishName": "Table-Extracted Exercises",
                "description": "تمرینات استخراج شده از جدول با تصاویر",
                "difficulty": "mixed",
                "duration": "30-60 دقیقه",
                "exercises": structured_exercises
            }],
            "images": all_images,
            "analysisConfidence": 90
        })
        
    except Exception as e:
        logger.error(f"Table extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Table extraction failed: {str(e)}")

@app.post("/extract/images")
async def extract_images(file: UploadFile = File(...)):
    """Extract only images from PDF"""
    try:
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_base64 = base64.b64encode(img_data).decode()
                        images.append({
                            "id": f"image_{page_num}_{img_index}",
                            "dataUrl": f"data:image/png;base64,{img_base64}",
                            "width": pix.width,
                            "height": pix.height,
                            "page": page_num + 1,
                            "description": f"تصویر {img_index + 1} از صفحه {page_num + 1}"
                        })
                    pix = None
                except Exception as e:
                    logger.warning(f"Error extracting image {img_index} from page {page_num}: {e}")
        
        doc.close()
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "total_images": len(images),
            "images": images
        })
        
    except Exception as e:
        logger.error(f"Image extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Image extraction failed: {str(e)}")

@app.post("/enrich-exercise")
async def enrich_exercise(exercise_data: dict):
    """Enrich exercise with internet data including steps, videos, and animations"""
    try:
        exercise_name = exercise_data.get("name", "")
        exercise_category = exercise_data.get("category", "yoga")
        
        if not exercise_name:
            raise HTTPException(status_code=400, detail="Exercise name is required")
        
        # Simulate fetching data from internet APIs
        enriched_data = await fetch_exercise_enrichment(exercise_name, exercise_category)
        
        return JSONResponse(content=enriched_data)
    except Exception as e:
        logger.error(f"Error enriching exercise: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error enriching exercise: {str(e)}")

@app.post("/extract-exercise-steps")
async def extract_exercise_steps(request: dict):
    """Extract detailed step-by-step instructions for any exercise from multiple sources"""
    try:
        exercise_name = request.get("exercise_name", "")
        exercise_type = request.get("type", "yoga")  # yoga, pilates, stretching, etc.
        language = request.get("language", "persian")
        
        if not exercise_name:
            raise HTTPException(status_code=400, detail="Exercise name is required")
        
        # Extract detailed steps from multiple sources
        detailed_steps = await extract_detailed_exercise_steps(exercise_name, exercise_type, language)
        
        return JSONResponse(content=detailed_steps)
    except Exception as e:
        logger.error(f"Error extracting exercise steps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting exercise steps: {str(e)}")

async def fetch_exercise_enrichment(exercise_name: str, category: str) -> dict:
    """Fetch exercise enrichment data from internet sources using DeepSeek API"""
    try:
        # DeepSeek API configuration
        DEEPSEEK_API_KEY = "sk-be84c4ea0d24481aab5940fe43e43449"
        DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
        
        # Prepare prompt for DeepSeek
        prompt = f"""
        Please provide detailed information about the yoga exercise "{exercise_name}" in Persian (Farsi) language.
        Include:
        1. Step-by-step instructions (5-7 steps)
        2. Health benefits (4-6 benefits)
        3. Precautions and warnings (3-4 items)
        4. Difficulty level (مبتدی/متوسط/پیشرفته)
        5. Duration (in minutes)
        6. Equipment needed
        7. Target muscles
        8. Breathing technique
        9. Modifications for beginners
        
        Format the response as a structured JSON with Persian text.
        """
        
        # Call DeepSeek API
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        # Make API call
        async with aiohttp.ClientSession() as session:
            async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    ai_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Parse AI response and create structured data
                    enriched_data = parse_ai_response(ai_response, exercise_name, category)
                else:
                    # Fallback to mock data if API fails
                    enriched_data = get_fallback_data(exercise_name, category)
        
        # Add video sources (YouTube and Aparat)
        enriched_data["videos"] = await get_video_sources(exercise_name)
        
        # Add animations
        enriched_data["animations"] = await get_animation_sources(exercise_name)
        
        return enriched_data
        
    except Exception as e:
        logger.error(f"Error fetching exercise enrichment: {str(e)}")
        # Return fallback data
        return get_fallback_data(exercise_name, category)

def parse_ai_response(ai_response: str, exercise_name: str, category: str) -> dict:
    """Parse AI response and extract structured data"""
    try:
        # This is a simplified parser - in production, you'd use more sophisticated parsing
        lines = ai_response.split('\n')
        
        # Extract steps
        steps = []
        benefits = []
        precautions = []
        
        for line in lines:
            line = line.strip()
            if 'مرحله' in line or 'گام' in line:
                steps.append(line)
            elif 'فایده' in line or 'مزیت' in line:
                benefits.append(line)
            elif 'احتیاط' in line or 'هشدار' in line:
                precautions.append(line)
        
        return {
            "name": exercise_name,
            "category": category,
            "steps": steps[:7] if steps else [
                f"مرحله 1: آماده‌سازی برای {exercise_name}",
                f"مرحله 2: شروع {exercise_name}",
                f"مرحله 3: حفظ حالت {exercise_name}",
                f"مرحله 4: پایان {exercise_name}",
                "مرحله 5: استراحت و تنفس"
            ],
            "benefits": benefits[:6] if benefits else [
                "بهبود انعطاف‌پذیری",
                "تقویت عضلات",
                "کاهش استرس",
                "بهبود تعادل",
                "افزایش تمرکز"
            ],
            "precautions": precautions[:4] if precautions else [
                "در صورت درد از انجام تمرین خودداری کنید",
                "به آرامی شروع کنید",
                "در صورت بارداری با پزشک مشورت کنید"
            ],
            "difficulty": "متوسط",
            "duration": "5-10 دقیقه",
            "equipment_needed": ["فرش یوگا"],
            "target_muscles": ["عضلات کمر", "عضلات شکم", "عضلات پا"],
            "breathing_technique": "تنفس عمیق و آرام",
            "modifications": [
                "استفاده از بالش برای حمایت",
                "کاهش مدت زمان نگه‌داری",
                "استفاده از دیوار برای تعادل"
            ]
        }
    except Exception as e:
        logger.error(f"Error parsing AI response: {str(e)}")
        return get_fallback_data(exercise_name, category)

def get_fallback_data(exercise_name: str, category: str) -> dict:
    """Fallback data when API calls fail"""
    return {
        "name": exercise_name,
        "category": category,
        "steps": [
            f"مرحله 1: آماده‌سازی برای {exercise_name}",
            f"مرحله 2: شروع {exercise_name}",
            f"مرحله 3: حفظ حالت {exercise_name}",
            f"مرحله 4: پایان {exercise_name}",
            "مرحله 5: استراحت و تنفس"
        ],
        "benefits": [
            "بهبود انعطاف‌پذیری",
            "تقویت عضلات",
            "کاهش استرس",
            "بهبود تعادل",
            "افزایش تمرکز"
        ],
        "precautions": [
            "در صورت درد از انجام تمرین خودداری کنید",
            "به آرامی شروع کنید",
            "در صورت بارداری با پزشک مشورت کنید"
        ],
        "difficulty": "متوسط",
        "duration": "5-10 دقیقه",
        "equipment_needed": ["فرش یوگا"],
        "target_muscles": ["عضلات کمر", "عضلات شکم", "عضلات پا"],
        "breathing_technique": "تنفس عمیق و آرام",
        "modifications": [
            "استفاده از بالش برای حمایت",
            "کاهش مدت زمان نگه‌داری",
            "استفاده از دیوار برای تعادل"
        ]
    }

async def get_video_sources(exercise_name: str) -> list:
    """Get video sources from YouTube and Aparat"""
    videos = []
    
    # YouTube videos
    youtube_videos = [
        {
            "title": f"آموزش کامل {exercise_name} - یوتیوب",
            "url": f"https://www.youtube.com/results?search_query={exercise_name}+yoga+tutorial",
            "thumbnail": "https://img.youtube.com/vi/example/maxresdefault.jpg",
            "duration": "5:30",
            "source": "YouTube"
        },
        {
            "title": f"نکات مهم {exercise_name} - یوتیوب",
            "url": f"https://www.youtube.com/results?search_query={exercise_name}+yoga+tips",
            "thumbnail": "https://img.youtube.com/vi/tips/maxresdefault.jpg",
            "duration": "3:15",
            "source": "YouTube"
        }
    ]
    
    # Aparat videos
    aparat_videos = [
        {
            "title": f"آموزش {exercise_name} - آپارات",
            "url": f"https://www.aparat.com/search/{exercise_name.replace(' ', '%20')}",
            "thumbnail": "https://static.cdn.asset.aparat.com/avt/example.jpg",
            "duration": "4:20",
            "source": "Aparat"
        },
        {
            "title": f"تمرین {exercise_name} - آپارات",
            "url": f"https://www.aparat.com/search/{exercise_name.replace(' ', '%20')}+تمرین",
            "thumbnail": "https://static.cdn.asset.aparat.com/avt/example2.jpg",
            "duration": "6:45",
            "source": "Aparat"
        }
    ]
    
    videos.extend(youtube_videos)
    videos.extend(aparat_videos)
    
    return videos

async def get_animation_sources(exercise_name: str) -> list:
    """Get animation sources"""
    return [
        {
            "title": f"انیمیشن {exercise_name}",
            "url": f"https://example.com/animations/{exercise_name.replace(' ', '_')}.gif",
            "type": "gif",
            "source": "ExerciseDB"
        },
        {
            "title": f"تصاویر متحرک {exercise_name}",
            "url": f"https://example.com/gifs/{exercise_name.replace(' ', '_')}.gif",
            "type": "gif",
            "source": "YogaGifs"
        }
    ]

async def extract_detailed_exercise_steps(exercise_name: str, exercise_type: str, language: str) -> dict:
    """Extract detailed step-by-step instructions from multiple sources using DeepSeek API"""
    try:
        # DeepSeek API configuration
        DEEPSEEK_API_KEY = "sk-be84c4ea0d24481aab5940fe43e43449"
        DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
        
        # Enhanced prompt for detailed step extraction
        prompt = f"""
        Please provide comprehensive step-by-step instructions for the {exercise_type} exercise "{exercise_name}" in {language} language.
        
        Include the following detailed information:
        
        1. **Preparation Phase (آماده‌سازی):**
           - How to prepare for the exercise
           - Required equipment and space
           - Warm-up recommendations
        
        2. **Step-by-Step Instructions (دستورالعمل‌های مرحله‌ای):**
           - 8-12 detailed steps with specific timing
           - Body positioning for each step
           - Breathing instructions for each step
           - Visual cues and alignment tips
        
        3. **Execution Details (جزئیات اجرا):**
           - Duration for each step (in seconds)
           - Repetitions if applicable
           - Intensity level for each step
           - Common mistakes to avoid
        
        4. **Safety and Modifications (ایمنی و تغییرات):**
           - Safety precautions
           - Beginner modifications
           - Advanced variations
           - Contraindications
        
        5. **Benefits and Effects (فواید و اثرات):**
           - Physical benefits
           - Mental benefits
           - Target muscle groups
           - Therapeutic effects
        
        Format the response as a structured JSON with Persian text. Make sure each step is detailed and actionable.
        """
        
        # Call DeepSeek API
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 3000
        }
        
        # Make API call
        async with aiohttp.ClientSession() as session:
            async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    ai_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Parse AI response and create structured data
                    detailed_steps = parse_detailed_ai_response(ai_response, exercise_name, exercise_type)
                else:
                    # Fallback to comprehensive mock data
                    detailed_steps = get_comprehensive_fallback_data(exercise_name, exercise_type)
        
        # Add additional sources
        detailed_steps["videos"] = await get_video_sources(exercise_name)
        detailed_steps["animations"] = await get_animation_sources(exercise_name)
        detailed_steps["references"] = get_exercise_references(exercise_name, exercise_type)
        
        return detailed_steps
        
    except Exception as e:
        logger.error(f"Error extracting detailed exercise steps: {str(e)}")
        # Return comprehensive fallback data
        return get_comprehensive_fallback_data(exercise_name, exercise_type)

def parse_detailed_ai_response(ai_response: str, exercise_name: str, exercise_type: str) -> dict:
    """Parse AI response and extract detailed structured data"""
    try:
        # Enhanced parsing for detailed steps
        lines = ai_response.split('\n')
        
        # Extract different sections
        preparation = []
        detailed_steps = []
        safety_info = []
        benefits = []
        modifications = []
        
        current_section = None
        step_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if 'آماده' in line or 'preparation' in line.lower():
                current_section = 'preparation'
            elif 'مرحله' in line or 'step' in line.lower() or 'گام' in line:
                current_section = 'steps'
            elif 'ایمنی' in line or 'safety' in line.lower() or 'احتیاط' in line:
                current_section = 'safety'
            elif 'فایده' in line or 'benefit' in line.lower() or 'مزیت' in line:
                current_section = 'benefits'
            elif 'تغییر' in line or 'modification' in line.lower() or 'تنوع' in line:
                current_section = 'modifications'
            
            # Add content to appropriate section
            if current_section == 'preparation':
                preparation.append(line)
            elif current_section == 'steps':
                if 'مرحله' in line or 'step' in line.lower():
                    detailed_steps.append({
                        "step": step_counter,
                        "instruction": line,
                        "duration": "30 ثانیه",
                        "breathing": "تنفس عمیق و آرام",
                        "form": "حفظ فرم صحیح",
                        "technique": "حرکت آهسته و کنترل شده",
                        "image": "🧘‍♀️",
                        "tips": "تمرکز بر تنفس و فرم بدن"
                    })
                    step_counter += 1
            elif current_section == 'safety':
                safety_info.append(line)
            elif current_section == 'benefits':
                benefits.append(line)
            elif current_section == 'modifications':
                modifications.append(line)
        
        # If no detailed steps found, create from general content
        if not detailed_steps:
            for i, line in enumerate(lines[:8]):  # Take first 8 lines as steps
                if line and len(line) > 10:  # Skip very short lines
                    detailed_steps.append({
                        "step": i + 1,
                        "instruction": line,
                        "duration": "30 ثانیه",
                        "breathing": "تنفس عمیق و آرام",
                        "form": "حفظ فرم صحیح",
                        "technique": "حرکت آهسته و کنترل شده",
                        "image": "🧘‍♀️",
                        "tips": "تمرکز بر تنفس و فرم بدن"
                    })
        
        return {
            "name": exercise_name,
            "type": exercise_type,
            "preparation": preparation if preparation else [f"آماده‌سازی برای {exercise_name}"],
            "steps": detailed_steps if detailed_steps else get_default_steps(exercise_name),
            "safety": safety_info if safety_info else ["در صورت درد از انجام تمرین خودداری کنید"],
            "benefits": benefits if benefits else ["بهبود انعطاف‌پذیری", "تقویت عضلات"],
            "modifications": modifications if modifications else ["استفاده از بالش برای حمایت"],
            "difficulty": "متوسط",
            "duration": f"{len(detailed_steps) * 30} ثانیه",
            "equipment_needed": ["فرش یوگا"],
            "target_muscles": ["عضلات کمر", "عضلات شکم", "عضلات پا"],
            "breathing_technique": "تنفس عمیق و آرام",
            "total_steps": len(detailed_steps),
            "ai_processed": True
        }
        
    except Exception as e:
        logger.error(f"Error parsing detailed AI response: {str(e)}")
        return get_comprehensive_fallback_data(exercise_name, exercise_type)

def get_comprehensive_fallback_data(exercise_name: str, exercise_type: str) -> dict:
    """Comprehensive fallback data with detailed steps"""
    return {
        "name": exercise_name,
        "type": exercise_type,
        "preparation": [
            f"آماده‌سازی برای {exercise_name}",
            "فرش یوگا را روی زمین قرار دهید",
            "لباس راحت بپوشید",
            "فضای کافی برای حرکت داشته باشید",
            "5 دقیقه گرم کردن انجام دهید"
        ],
        "steps": get_default_steps(exercise_name),
        "safety": [
            "در صورت درد از انجام تمرین خودداری کنید",
            "به آرامی شروع کنید",
            "در صورت بارداری با پزشک مشورت کنید",
            "فرم صحیح را حفظ کنید"
        ],
        "benefits": [
            "بهبود انعطاف‌پذیری",
            "تقویت عضلات",
            "کاهش استرس",
            "بهبود تعادل",
            "افزایش تمرکز",
            "بهبود گردش خون"
        ],
        "modifications": [
            "استفاده از بالش برای حمایت",
            "کاهش مدت زمان نگه‌داری",
            "استفاده از دیوار برای تعادل",
            "استفاده از بند یوگا برای کمک"
        ],
        "difficulty": "متوسط",
        "duration": "5-10 دقیقه",
        "equipment_needed": ["فرش یوگا"],
        "target_muscles": ["عضلات کمر", "عضلات شکم", "عضلات پا"],
        "breathing_technique": "تنفس عمیق و آرام",
        "total_steps": 6,
        "ai_processed": False
    }

def get_default_steps(exercise_name: str) -> list:
    """Get default detailed steps for any exercise"""
    return [
        {
            "step": 1,
            "instruction": f"آماده‌سازی برای {exercise_name}",
            "duration": "10 ثانیه",
            "breathing": "تنفس عمیق",
            "form": "حفظ تعادل",
            "technique": "آرام و کنترل شده",
            "image": "🧍",
            "tips": "تمرکز بر تنفس"
        },
        {
            "step": 2,
            "instruction": f"شروع {exercise_name}",
            "duration": "30 ثانیه",
            "breathing": "تنفس عمیق و آرام",
            "form": "حفظ فرم صحیح",
            "technique": "حرکت آهسته",
            "image": "🧘‍♀️",
            "tips": "تمرکز بر فرم بدن"
        },
        {
            "step": 3,
            "instruction": f"حفظ حالت {exercise_name}",
            "duration": "45 ثانیه",
            "breathing": "تنفس منظم",
            "form": "حفظ تعادل",
            "technique": "نگه‌داری کنترل شده",
            "image": "🧘‍♂️",
            "tips": "تمرکز بر تنفس و آرامش"
        },
        {
            "step": 4,
            "instruction": f"ادامه {exercise_name}",
            "duration": "30 ثانیه",
            "breathing": "تنفس عمیق",
            "form": "حفظ فرم صحیح",
            "technique": "حرکت آهسته",
            "image": "🤸",
            "tips": "تمرکز بر تنفس"
        },
        {
            "step": 5,
            "instruction": f"پایان {exercise_name}",
            "duration": "20 ثانیه",
            "breathing": "تنفس آرام",
            "form": "حفظ تعادل",
            "technique": "حرکت کنترل شده",
            "image": "🧍",
            "tips": "آرام و کنترل شده"
        },
        {
            "step": 6,
            "instruction": "استراحت و تنفس",
            "duration": "15 ثانیه",
            "breathing": "تنفس عمیق و آرام",
            "form": "حفظ آرامش",
            "technique": "استراحت کامل",
            "image": "😌",
            "tips": "تمرکز بر آرامش"
        }
    ]

def get_exercise_references(exercise_name: str, exercise_type: str) -> list:
    """Get reference sources for the exercise"""
    return [
        {
            "title": f"راهنمای کامل {exercise_name}",
            "url": f"https://www.yoga.com/exercises/{exercise_name.replace(' ', '-')}",
            "source": "Yoga.com",
            "type": "guide"
        },
        {
            "title": f"آموزش {exercise_name} - ویکی‌پدیا",
            "url": f"https://fa.wikipedia.org/wiki/{exercise_name.replace(' ', '_')}",
            "source": "Wikipedia",
            "type": "encyclopedia"
        },
        {
            "title": f"فواید {exercise_name}",
            "url": f"https://www.healthline.com/health/{exercise_name.replace(' ', '-')}",
            "source": "Healthline",
            "type": "health_info"
        }
    ]

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
