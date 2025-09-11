#!/usr/bin/env python3
"""
Database module for Yoga Persian Assistant
Handles all database operations for exercises, PDFs, and user data
"""

import sqlite3
import json
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class YogaDatabase:
    def __init__(self, db_path: str = "yoga_assistant.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create exercises table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS exercises (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        persian_name TEXT,
                        category TEXT,
                        difficulty TEXT,
                        duration INTEGER,
                        equipment TEXT,
                        muscles TEXT,
                        description TEXT,
                        benefits TEXT,
                        precautions TEXT,
                        breathing_technique TEXT,
                        modifications TEXT,
                        image_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create exercise_steps table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS exercise_steps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        exercise_id INTEGER,
                        step_number INTEGER,
                        title TEXT,
                        description TEXT,
                        duration INTEGER,
                        image_data TEXT,
                        FOREIGN KEY (exercise_id) REFERENCES exercises (id) ON DELETE CASCADE
                    )
                ''')
                
                # Create pdf_extractions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS pdf_extractions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        file_size INTEGER,
                        extraction_method TEXT,
                        extracted_text TEXT,
                        extracted_images TEXT,
                        extracted_tables TEXT,
                        processing_time REAL,
                        confidence_score REAL,
                        status TEXT DEFAULT 'completed',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create exercise_videos table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS exercise_videos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        exercise_id INTEGER,
                        title TEXT,
                        url TEXT,
                        source TEXT,
                        duration INTEGER,
                        thumbnail_url TEXT,
                        FOREIGN KEY (exercise_id) REFERENCES exercises (id) ON DELETE CASCADE
                    )
                ''')
                
                # Create exercise_animations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS exercise_animations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        exercise_id INTEGER,
                        title TEXT,
                        url TEXT,
                        source TEXT,
                        type TEXT,
                        FOREIGN KEY (exercise_id) REFERENCES exercises (id) ON DELETE CASCADE
                    )
                ''')
                
                # Create user_progress table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_progress (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        exercise_id INTEGER,
                        session_date DATE,
                        duration_completed INTEGER,
                        total_duration INTEGER,
                        completed_steps INTEGER,
                        total_steps INTEGER,
                        notes TEXT,
                        rating INTEGER,
                        FOREIGN KEY (exercise_id) REFERENCES exercises (id) ON DELETE CASCADE
                    )
                ''')
                
                # Create user_favorites table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_favorites (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        exercise_id INTEGER,
                        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (exercise_id) REFERENCES exercises (id) ON DELETE CASCADE
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_exercises_category ON exercises(category)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_exercises_difficulty ON exercises(difficulty)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_pdf_extractions_filename ON pdf_extractions(filename)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_progress_date ON user_progress(session_date)')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_exercise(self, exercise_data: Dict[str, Any]) -> int:
        """Save an exercise to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if exercise already exists
                cursor.execute('SELECT id FROM exercises WHERE name = ?', (exercise_data.get('name'),))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing exercise
                    cursor.execute('''
                        UPDATE exercises SET
                            persian_name = ?, category = ?, difficulty = ?, duration = ?,
                            equipment = ?, muscles = ?, description = ?, benefits = ?,
                            precautions = ?, breathing_technique = ?, modifications = ?,
                            image_data = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE name = ?
                    ''', (
                        exercise_data.get('persian_name'),
                        exercise_data.get('category'),
                        exercise_data.get('difficulty'),
                        exercise_data.get('duration'),
                        json.dumps(exercise_data.get('equipment', [])),
                        json.dumps(exercise_data.get('muscles', [])),
                        exercise_data.get('description'),
                        exercise_data.get('benefits'),
                        exercise_data.get('precautions'),
                        exercise_data.get('breathing_technique'),
                        exercise_data.get('modifications'),
                        exercise_data.get('image_data'),
                        exercise_data.get('name')
                    ))
                    exercise_id = existing[0]
                else:
                    # Insert new exercise
                    cursor.execute('''
                        INSERT INTO exercises (
                            name, persian_name, category, difficulty, duration,
                            equipment, muscles, description, benefits, precautions,
                            breathing_technique, modifications, image_data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        exercise_data.get('name'),
                        exercise_data.get('persian_name'),
                        exercise_data.get('category'),
                        exercise_data.get('difficulty'),
                        exercise_data.get('duration'),
                        json.dumps(exercise_data.get('equipment', [])),
                        json.dumps(exercise_data.get('muscles', [])),
                        exercise_data.get('description'),
                        exercise_data.get('benefits'),
                        exercise_data.get('precautions'),
                        exercise_data.get('breathing_technique'),
                        exercise_data.get('modifications'),
                        exercise_data.get('image_data')
                    ))
                    exercise_id = cursor.lastrowid
                
                conn.commit()
                return exercise_id
                
        except Exception as e:
            logger.error(f"Error saving exercise: {e}")
            raise
    
    def save_exercise_steps(self, exercise_id: int, steps: List[Dict[str, Any]]):
        """Save exercise steps to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete existing steps
                cursor.execute('DELETE FROM exercise_steps WHERE exercise_id = ?', (exercise_id,))
                
                # Insert new steps
                for step in steps:
                    cursor.execute('''
                        INSERT INTO exercise_steps (
                            exercise_id, step_number, title, description, duration, image_data
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        exercise_id,
                        step.get('step_number'),
                        step.get('title'),
                        step.get('description'),
                        step.get('duration'),
                        step.get('image_data')
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving exercise steps: {e}")
            raise
    
    def save_pdf_extraction(self, extraction_data: Dict[str, Any]) -> int:
        """Save PDF extraction data to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO pdf_extractions (
                        filename, file_size, extraction_method, extracted_text,
                        extracted_images, extracted_tables, processing_time,
                        confidence_score, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    extraction_data.get('filename'),
                    extraction_data.get('file_size'),
                    extraction_data.get('extraction_method'),
                    extraction_data.get('extracted_text'),
                    json.dumps(extraction_data.get('extracted_images', [])),
                    json.dumps(extraction_data.get('extracted_tables', [])),
                    extraction_data.get('processing_time'),
                    extraction_data.get('confidence_score'),
                    extraction_data.get('status', 'completed')
                ))
                
                extraction_id = cursor.lastrowid
                conn.commit()
                return extraction_id
                
        except Exception as e:
            logger.error(f"Error saving PDF extraction: {e}")
            raise
    
    def get_exercises(self, category: Optional[str] = None, difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get exercises from the database with optional filters"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = 'SELECT * FROM exercises WHERE 1=1'
                params = []
                
                if category:
                    query += ' AND category = ?'
                    params.append(category)
                
                if difficulty:
                    query += ' AND difficulty = ?'
                    params.append(difficulty)
                
                query += ' ORDER BY created_at DESC'
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                exercises = []
                for row in rows:
                    exercise = dict(row)
                    # Parse JSON fields
                    exercise['equipment'] = json.loads(exercise['equipment']) if exercise['equipment'] else []
                    exercise['muscles'] = json.loads(exercise['muscles']) if exercise['muscles'] else []
                    exercises.append(exercise)
                
                return exercises
                
        except Exception as e:
            logger.error(f"Error getting exercises: {e}")
            return []
    
    def get_exercise_by_id(self, exercise_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific exercise by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM exercises WHERE id = ?', (exercise_id,))
                row = cursor.fetchone()
                
                if row:
                    exercise = dict(row)
                    exercise['equipment'] = json.loads(exercise['equipment']) if exercise['equipment'] else []
                    exercise['muscles'] = json.loads(exercise['muscles']) if exercise['muscles'] else []
                    return exercise
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting exercise by ID: {e}")
            return None
    
    def get_exercise_steps(self, exercise_id: int) -> List[Dict[str, Any]]:
        """Get steps for a specific exercise"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM exercise_steps 
                    WHERE exercise_id = ? 
                    ORDER BY step_number
                ''', (exercise_id,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting exercise steps: {e}")
            return []
    
    def save_exercise_videos(self, exercise_id: int, videos: List[Dict[str, Any]]):
        """Save exercise videos to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete existing videos
                cursor.execute('DELETE FROM exercise_videos WHERE exercise_id = ?', (exercise_id,))
                
                # Insert new videos
                for video in videos:
                    cursor.execute('''
                        INSERT INTO exercise_videos (
                            exercise_id, title, url, source, duration, thumbnail_url
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        exercise_id,
                        video.get('title'),
                        video.get('url'),
                        video.get('source'),
                        video.get('duration'),
                        video.get('thumbnail_url')
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving exercise videos: {e}")
            raise
    
    def save_exercise_animations(self, exercise_id: int, animations: List[Dict[str, Any]]):
        """Save exercise animations to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete existing animations
                cursor.execute('DELETE FROM exercise_animations WHERE exercise_id = ?', (exercise_id,))
                
                # Insert new animations
                for animation in animations:
                    cursor.execute('''
                        INSERT INTO exercise_animations (
                            exercise_id, title, url, source, type
                        ) VALUES (?, ?, ?, ?, ?)
                    ''', (
                        exercise_id,
                        animation.get('title'),
                        animation.get('url'),
                        animation.get('source'),
                        animation.get('type')
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving exercise animations: {e}")
            raise
    
    def save_user_progress(self, progress_data: Dict[str, Any]) -> int:
        """Save user progress for an exercise session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO user_progress (
                        exercise_id, session_date, duration_completed, total_duration,
                        completed_steps, total_steps, notes, rating
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    progress_data.get('exercise_id'),
                    progress_data.get('session_date'),
                    progress_data.get('duration_completed'),
                    progress_data.get('total_duration'),
                    progress_data.get('completed_steps'),
                    progress_data.get('total_steps'),
                    progress_data.get('notes'),
                    progress_data.get('rating')
                ))
                
                progress_id = cursor.lastrowid
                conn.commit()
                return progress_id
                
        except Exception as e:
            logger.error(f"Error saving user progress: {e}")
            raise
    
    def add_to_favorites(self, exercise_id: int) -> bool:
        """Add an exercise to user favorites"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if already in favorites
                cursor.execute('SELECT id FROM user_favorites WHERE exercise_id = ?', (exercise_id,))
                if cursor.fetchone():
                    return False  # Already in favorites
                
                cursor.execute('INSERT INTO user_favorites (exercise_id) VALUES (?)', (exercise_id,))
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error adding to favorites: {e}")
            return False
    
    def remove_from_favorites(self, exercise_id: int) -> bool:
        """Remove an exercise from user favorites"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM user_favorites WHERE exercise_id = ?', (exercise_id,))
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error removing from favorites: {e}")
            return False
    
    def get_favorites(self) -> List[Dict[str, Any]]:
        """Get user's favorite exercises"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT e.*, uf.added_at as favorited_at
                    FROM exercises e
                    INNER JOIN user_favorites uf ON e.id = uf.exercise_id
                    ORDER BY uf.added_at DESC
                ''')
                
                rows = cursor.fetchall()
                exercises = []
                for row in rows:
                    exercise = dict(row)
                    exercise['equipment'] = json.loads(exercise['equipment']) if exercise['equipment'] else []
                    exercise['muscles'] = json.loads(exercise['muscles']) if exercise['muscles'] else []
                    exercises.append(exercise)
                
                return exercises
                
        except Exception as e:
            logger.error(f"Error getting favorites: {e}")
            return []
    
    def get_user_progress(self, exercise_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get user progress data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if exercise_id:
                    cursor.execute('''
                        SELECT up.*, e.name as exercise_name
                        FROM user_progress up
                        INNER JOIN exercises e ON up.exercise_id = e.id
                        WHERE up.exercise_id = ?
                        ORDER BY up.session_date DESC
                    ''', (exercise_id,))
                else:
                    cursor.execute('''
                        SELECT up.*, e.name as exercise_name
                        FROM user_progress up
                        INNER JOIN exercises e ON up.exercise_id = e.id
                        ORDER BY up.session_date DESC
                    ''')
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting user progress: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count exercises
                cursor.execute('SELECT COUNT(*) FROM exercises')
                stats['total_exercises'] = cursor.fetchone()[0]
                
                # Count by category
                cursor.execute('SELECT category, COUNT(*) FROM exercises GROUP BY category')
                stats['exercises_by_category'] = dict(cursor.fetchall())
                
                # Count by difficulty
                cursor.execute('SELECT difficulty, COUNT(*) FROM exercises GROUP BY difficulty')
                stats['exercises_by_difficulty'] = dict(cursor.fetchall())
                
                # Count PDF extractions
                cursor.execute('SELECT COUNT(*) FROM pdf_extractions')
                stats['total_pdf_extractions'] = cursor.fetchone()[0]
                
                # Count user progress sessions
                cursor.execute('SELECT COUNT(*) FROM user_progress')
                stats['total_progress_sessions'] = cursor.fetchone()[0]
                
                # Count favorites
                cursor.execute('SELECT COUNT(*) FROM user_favorites')
                stats['total_favorites'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

# Global database instance
db = YogaDatabase()
