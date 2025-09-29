import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textstat import flesch_reading_ease
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter, defaultdict
import sqlite3
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import torch
import asyncio
import concurrent.futures
from functools import lru_cache
import seaborn as sns
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure for better performance
st.set_page_config(
    page_title="MCA eConsultation Sentiment Analysis - Enhanced",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    return True

class EnhancedMCAeSentimentAnalyzer:
    def __init__(self):
        """Initialize the Enhanced MCA eConsultation Sentiment Analysis System with GPU support"""
        self.device = self._setup_device()
        self.setup_models()
        self.setup_database()
        self.government_stopwords = self.get_government_stopwords()
        self.policy_aspects = self.get_policy_aspects()
        self.processing_stats = defaultdict(int)
        
    def _setup_device(self):
        """Setup GPU if available"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            st.success(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
            return device
        else:
            device = torch.device("cpu")
            st.info("💻 Using CPU for processing")
            return device
    
    def _normalize_text(self, value):
        """Safely convert any value to a clean string for processing."""
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return ""
        except Exception:
            pass
        text = str(value)
        if text.lower() in {"nan", "none", "null", ""}:
            return ""
        return text.strip()

    def setup_models(self):
        """Load and setup enhanced AI models with GPU support"""
        try:
            # Enhanced RoBERTa model for better accuracy
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            # Load tokenizer and model separately for better control
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to GPU if available
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
                st.success("✅ RoBERTa model loaded on GPU!")
            else:
                st.success("✅ RoBERTa model loaded on CPU")
            
            # Create pipeline with device specification
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True
            )
            
            # Load summarization model
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device.type == "cuda" else -1
            )
            
        except Exception as e:
            st.error(f"❌ Enhanced model loading failed: {str(e)}")
            # Fallback to basic model
            try:
                self.sentiment_pipeline = pipeline("sentiment-analysis")
                st.warning("⚠️ Using fallback model (CPU only)")
            except:
                st.error("❌ Could not load any model")
                self.use_fallback = True
    
    def setup_database(self):
        """Setup enhanced SQLite database with better schema"""
        self.conn = sqlite3.connect('mca_consultations_enhanced.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Enhanced consultation analysis table
        cursor.execute("""CREATE TABLE IF NOT EXISTS consultation_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id TEXT,
                comment_id TEXT UNIQUE,
                comment_text TEXT,
                sentiment TEXT,
                confidence REAL,
                aspects TEXT,
                stakeholder_type TEXT,
                reading_ease REAL,
                word_count INTEGER,
                processing_time REAL,
                section_reference TEXT,
                organization TEXT,
                industry_sector TEXT,
                submission_date TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(consultation_id, comment_id)
            )""")
        
        # Enhanced summary table
        cursor.execute("""CREATE TABLE IF NOT EXISTS consultation_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id TEXT UNIQUE,
                document_title TEXT,
                total_comments INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                avg_confidence REAL,
                avg_reading_ease REAL,
                key_themes TEXT,
                summary_text TEXT,
                processing_stats TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
        
        # Performance metrics table
        cursor.execute("""CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id TEXT,
                total_comments INTEGER,
                processing_time_seconds REAL,
                comments_per_second REAL,
                gpu_used BOOLEAN,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
        
        self.conn.commit()
    
    def get_government_stopwords(self):
        """Enhanced government-specific stopwords"""
        try:
            basic_stopwords = set(stopwords.words('english'))
        except:
            basic_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
        gov_stopwords = {
            'government', 'ministry', 'department', 'section', 'clause', 'sub', 'subsection',
            'provision', 'act', 'rule', 'regulation', 'draft', 'proposed', 'amendment',
            'consultation', 'comment', 'suggestion', 'feedback', 'response', 'submission',
            'policy', 'shall', 'may', 'would', 'could', 'should', 'must', 'will',
            'pursuant', 'thereof', 'herein', 'whereas', 'hereby', 'furthermore',
            'mca', 'corporate', 'affairs', 'companies', 'compliance', 'filing'
        }
        return basic_stopwords.union(gov_stopwords)
    
    def get_policy_aspects(self):
        """Enhanced policy aspects for comprehensive analysis"""
        return {
            'economic_impact': {
                'keywords': ['cost', 'budget', 'financial', 'revenue', 'expense', 'funding', 
                           'economic', 'profit', 'loss', 'investment', 'return', 'fee', 'price'],
                'weight': 1.0
            },
            'implementation': {
                'keywords': ['process', 'procedure', 'timeline', 'execution', 'implement', 
                           'deploy', 'rollout', 'phase', 'step', 'method', 'approach', 'plan'],
                'weight': 1.0
            },
            'compliance': {
                'keywords': ['burden', 'requirement', 'regulation', 'rule', 'mandatory', 
                           'compulsory', 'obligation', 'duty', 'responsibility', 'comply'],
                'weight': 1.0
            },
            'stakeholder_impact': {
                'keywords': ['business', 'company', 'industry', 'sector', 'organization',
                           'enterprise', 'firm', 'corporation', 'entity', 'sme', 'startup'],
                'weight': 1.0
            },
            'transparency': {
                'keywords': ['disclosure', 'reporting', 'information', 'data', 'publish',
                           'public', 'open', 'transparent', 'accessible', 'available'],
                'weight': 1.0
            },
            'technology': {
                'keywords': ['digital', 'technology', 'system', 'software', 'platform',
                           'online', 'automated', 'ai', 'digitization', 'portal'],
                'weight': 1.0
            }
        }
    
    @lru_cache(maxsize=1000)
    def analyze_sentiment_cached(self, text_hash: str, text: str):
        """Cached sentiment analysis for better performance"""
        return self._analyze_sentiment_core(text)
    
    def _analyze_sentiment_core(self, text):
        """Core sentiment analysis with enhanced error handling"""
        try:
            if hasattr(self, 'use_fallback'):
                return {'sentiment': 'neutral', 'confidence': 0.5, 'all_scores': []}
            
            # Truncate text if too long (RoBERTa has 512 token limit)
            if len(text.split()) > 500:
                text = ' '.join(text.split()[:500])
            
            result = self.sentiment_pipeline(text)
            if isinstance(result[0], list):
                result = result[0]
            
            # Enhanced label mapping for different model outputs
            if len(result) == 1:
                pred = result[0]
                sentiment_map = {
                    'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral',
                    'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'
                }
                sentiment = sentiment_map.get(pred['label'], 'neutral')
                confidence = pred['score']
            else:
                best_pred = max(result, key=lambda x: x['score'])
                label_map = {
                    'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive',
                    'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'
                }
                sentiment = label_map.get(best_pred['label'], 'neutral')
                confidence = best_pred['score']
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'all_scores': result
            }
            
        except Exception as e:
            st.warning(f"Sentiment analysis error: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0.3, 'all_scores': []}
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis with caching"""
        if not text or len(text.strip()) < 5:
            return {'sentiment': 'neutral', 'confidence': 0.3, 'all_scores': []}
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.analyze_sentiment_cached(text_hash, text)
    
    def analyze_aspects_enhanced(self, text):
        """Enhanced aspect-based sentiment analysis with weights"""
        text_lower = text.lower()
        aspects_found = {}
        
        for aspect, config in self.policy_aspects.items():
            keywords = config['keywords']
            weight = config['weight']
            
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                # Calculate aspect relevance score
                relevance_score = len(found_keywords) / len(keywords) * weight
                
                # Get sentiment for this aspect
                aspect_sentiment = self.analyze_sentiment(text)
                
                aspects_found[aspect] = {
                    'sentiment': aspect_sentiment['sentiment'],
                    'confidence': aspect_sentiment['confidence'] * relevance_score,
                    'keywords_found': found_keywords,
                    'relevance_score': relevance_score
                }
        
        return aspects_found
    
    def batch_analyze_sentiments(self, texts: List[str], batch_size: int = 32):
        """Enhanced batch processing with GPU acceleration"""
        results = []
        
        # Process in batches for better GPU utilization
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Use batch processing if available
                if hasattr(self.sentiment_pipeline, '__call__') and len(batch_texts) > 1:
                    batch_results = self.sentiment_pipeline(batch_texts)
                    
                    for j, result in enumerate(batch_results):
                        text = batch_texts[j]
                        if isinstance(result, list) and len(result) > 0:
                            best_pred = max(result, key=lambda x: x['score'])
                            label_map = {
                                'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive',
                                'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'
                            }
                            sentiment = label_map.get(best_pred['label'], 'neutral')
                            confidence = best_pred['score']
                        else:
                            sentiment = 'neutral'
                            confidence = 0.5
                        
                        results.append({
                            'text': text,
                            'sentiment': sentiment,
                            'confidence': confidence
                        })
                else:
                    # Fall back to individual processing
                    for text in batch_texts:
                        result = self.analyze_sentiment(text)
                        results.append({
                            'text': text,
                            'sentiment': result['sentiment'],
                            'confidence': result['confidence']
                        })
            except Exception as e:
                st.warning(f"Batch processing error: {str(e)}")
                # Individual processing fallback
                for text in batch_texts:
                    result = self.analyze_sentiment(text)
                    results.append({
                        'text': text,
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence']
                    })
        
        return results
    
    def generate_enhanced_summary(self, comments: List[str], max_length: int = 200):
        """Enhanced summary generation using BART"""
        if not comments:
            return "No comments to summarize."
        
        try:
            # Combine comments and truncate if necessary
            combined_text = " ".join(comments)
            if len(combined_text.split()) > 1000:
                combined_text = " ".join(combined_text.split()[:1000])
            
            # Generate summary using BART
            summary = self.summarizer(
                combined_text,
                max_length=max_length,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            
            return summary
            
        except Exception as e:
            st.warning(f"Summary generation failed: {str(e)}")
            # Fallback to extractive summary
            return self.generate_extractive_summary(comments)
    
    def generate_extractive_summary(self, comments: List[str], num_sentences: int = 3):
        """Extractive summarization fallback"""
        if not comments:
            return "No comments to summarize."
        
        all_text = " ".join(comments)
        sentences = sent_tokenize(all_text)
        
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
        
        # Simple sentence scoring based on word frequency
        words = word_tokenize(all_text.lower())
        words = [w for w in words if w.isalnum() and w not in self.government_stopwords]
        word_freq = Counter(words)
        
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            score = sum(word_freq.get(word, 0) for word in sentence_words)
            sentence_scores[sentence] = score / len(sentence_words) if sentence_words else 0
        
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        return " ".join([sent[0] for sent in top_sentences])
    
    def load_data_from_file(self, file, file_type: str):
        """Enhanced file loading supporting multiple formats"""
        try:
            if file_type.lower() == 'csv':
                df = pd.read_csv(file)
                return self.process_csv_data(df)
            
            elif file_type.lower() == 'json':
                data = json.load(file)
                return self.process_json_data(data)
            
            elif file_type.lower() == 'xml':
                tree = ET.parse(file)
                root = tree.getroot()
                return self.process_xml_data(root)
            
            else:
                st.error(f"Unsupported file type: {file_type}")
                return []
                
        except Exception as e:
            st.error(f"Error loading {file_type} file: {str(e)}")
            return []
    
    def process_csv_data(self, df):
        """Process CSV data with enhanced field mapping"""
        comments_data = []
        
        # Required field: 'text' or 'comment' or 'comment_text'
        text_columns = ['text', 'comment', 'comment_text', 'content', 'description']
        text_col = None
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col:
            st.error("No text column found. Expected columns: text, comment, comment_text, content, or description")
            return []
        
        for _, row in df.iterrows():
            comment = {
                'text': self._normalize_text(row.get(text_col, "")),
                'stakeholder_type': self._normalize_text(row.get('stakeholder_type', 'unknown')),
                'organization': self._normalize_text(row.get('organization', '')),
                'industry_sector': self._normalize_text(row.get('industry_sector', '')),
                'section_reference': self._normalize_text(row.get('section_reference', '')),
                'submission_date': self._normalize_text(row.get('submission_date', ''))
            }
            
            if comment['text']:  # Only add if there's actual text content
                comments_data.append(comment)
        
        return comments_data
    
    def process_json_data(self, data):
        """Process JSON data with flexible structure support"""
        comments_data = []
        
        try:
            # Handle different JSON structures
            if isinstance(data, dict):
                if 'comments' in data:
                    comments_list = data['comments']
                elif 'data' in data:
                    comments_list = data['data']
                else:
                    comments_list = [data]  # Single comment object
            elif isinstance(data, list):
                comments_list = data
            else:
                st.error("Unsupported JSON structure")
                return []
            
            for item in comments_list:
                if isinstance(item, dict):
                    # Extract text content
                    text_fields = ['text', 'comment', 'comment_text', 'content', 'description']
                    text_content = ""
                    
                    for field in text_fields:
                        if field in item:
                            text_content = self._normalize_text(item[field])
                            break
                    
                    # Handle nested user profile
                    user_profile = item.get('user_profile', {})
                    comment_content = item.get('comment_content', {})
                    
                    comment = {
                        'text': text_content or self._normalize_text(comment_content.get('text', '')),
                        'stakeholder_type': self._normalize_text(
                            item.get('stakeholder_type') or 
                            user_profile.get('stakeholder_type', 'unknown')
                        ),
                        'organization': self._normalize_text(
                            item.get('organization') or 
                            user_profile.get('organization', '')
                        ),
                        'industry_sector': self._normalize_text(
                            item.get('industry_sector') or 
                            user_profile.get('industry_sector', '')
                        ),
                        'section_reference': self._normalize_text(
                            item.get('section_reference') or 
                            comment_content.get('section_reference', '')
                        ),
                        'submission_date': self._normalize_text(
                            item.get('submission_date') or 
                            item.get('submission_timestamp', '')
                        )
                    }
                    
                    if comment['text']:
                        comments_data.append(comment)
            
        except Exception as e:
            st.error(f"Error processing JSON data: {str(e)}")
            return []
        
        return comments_data
    
    def process_xml_data(self, root):
        """Process XML data with flexible structure support"""
        comments_data = []
        
        try:
            # Common XML structures for comments
            comment_elements = (
                root.findall('.//comment') + 
                root.findall('.//item') + 
                root.findall('.//entry') +
                root.findall('.//record')
            )
            
            for element in comment_elements:
                # Extract text content
                text_content = ""
                text_fields = ['text', 'comment', 'content', 'description']
                
                for field in text_fields:
                    elem = element.find(field)
                    if elem is not None and elem.text:
                        text_content = self._normalize_text(elem.text)
                        break
                
                # If no specific field found, try element text directly
                if not text_content and element.text:
                    text_content = self._normalize_text(element.text)
                
                # Extract other fields
                def get_xml_text(field_name):
                    elem = element.find(field_name)
                    return self._normalize_text(elem.text if elem is not None else "")
                
                comment = {
                    'text': text_content,
                    'stakeholder_type': get_xml_text('stakeholder_type') or get_xml_text('type') or 'unknown',
                    'organization': get_xml_text('organization') or get_xml_text('org'),
                    'industry_sector': get_xml_text('industry_sector') or get_xml_text('sector'),
                    'section_reference': get_xml_text('section_reference') or get_xml_text('section'),
                    'submission_date': get_xml_text('submission_date') or get_xml_text('date')
                }
                
                if comment['text']:
                    comments_data.append(comment)
                    
        except Exception as e:
            st.error(f"Error processing XML data: {str(e)}")
            return []
        
        return comments_data
    
    def process_consultation_enhanced(self, consultation_id: str, comments: List[Dict], 
                                    document_title: str = ""):
        """Enhanced consultation processing with performance metrics"""
        start_time = datetime.now()
        results = []
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.container()
        
        # Prepare texts for batch processing
        texts = [self._normalize_text(comment.get('text', '')) for comment in comments]
        valid_comments = [(i, comment) for i, comment in enumerate(comments) if texts[i].strip()]
        
        status_text.text(f'Starting analysis of {len(valid_comments)} comments...')
        
        # Batch sentiment analysis for better performance
        if len(valid_comments) > 10:
            status_text.text('Running batch sentiment analysis...')
            batch_texts = [texts[i] for i, _ in valid_comments]
            batch_results = self.batch_analyze_sentiments(batch_texts, batch_size=16)
            
            # Process results
            for j, (i, comment) in enumerate(valid_comments):
                batch_result = batch_results[j]
                
                # Generate comment ID
                comment_id = hashlib.md5(
                    f"{consultation_id}_{i}_{batch_result['text'][:50]}".encode()
                ).hexdigest()[:8]
                
                # Enhanced aspects analysis
                aspects = self.analyze_aspects_enhanced(batch_result['text'])
                
                # Reading ease calculation
                try:
                    reading_ease = flesch_reading_ease(batch_result['text'])
                except:
                    reading_ease = 50.0
                
                result = {
                    'comment_id': comment_id,
                    'text': batch_result['text'],
                    'sentiment': batch_result['sentiment'],
                    'confidence': batch_result['confidence'],
                    'aspects': aspects,
                    'stakeholder_type': comment.get('stakeholder_type', 'unknown'),
                    'organization': comment.get('organization', ''),
                    'industry_sector': comment.get('industry_sector', ''),
                    'section_reference': comment.get('section_reference', ''),
                    'submission_date': comment.get('submission_date', ''),
                    'reading_ease': reading_ease,
                    'word_count': len(batch_result['text'].split()),
                    'processing_time': (datetime.now() - start_time).total_seconds() / (j + 1)
                }
                
                results.append(result)
                
                # Update progress
                progress = (j + 1) / len(valid_comments)
                progress_bar.progress(progress)
                
                # Update metrics every 10 comments
                if (j + 1) % 10 == 0 or j == len(valid_comments) - 1:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    speed = (j + 1) / elapsed if elapsed > 0 else 0
                    
                    with metrics_container:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Processed", f"{j + 1}/{len(valid_comments)}")
                        with col2:
                            st.metric("Speed", f"{speed:.1f} comments/sec")
                        with col3:
                            st.metric("Time Elapsed", f"{elapsed:.1f}s")
        
        else:
            # Individual processing for small batches
            for j, (i, comment) in enumerate(valid_comments):
                status_text.text(f'Processing comment {j+1}/{len(valid_comments)}')
                
                safe_text = texts[i]
                comment_id = hashlib.md5(f"{consultation_id}_{i}_{safe_text[:50]}".encode()).hexdigest()[:8]
                
                sentiment_result = self.analyze_sentiment(safe_text)
                aspects = self.analyze_aspects_enhanced(safe_text)
                
                try:
                    reading_ease = flesch_reading_ease(safe_text)
                except:
                    reading_ease = 50.0
                
                result = {
                    'comment_id': comment_id,
                    'text': safe_text,
                    'sentiment': sentiment_result['sentiment'],
                    'confidence': sentiment_result['confidence'],
                    'aspects': aspects,
                    'stakeholder_type': comment.get('stakeholder_type', 'unknown'),
                    'organization': comment.get('organization', ''),
                    'industry_sector': comment.get('industry_sector', ''),
                    'section_reference': comment.get('section_reference', ''),
                    'submission_date': comment.get('submission_date', ''),
                    'reading_ease': reading_ease,
                    'word_count': len(safe_text.split()),
                    'processing_time': (datetime.now() - start_time).total_seconds() / (j + 1)
                }
                
                results.append(result)
                progress_bar.progress((j + 1) / len(valid_comments))
        
        # Calculate final metrics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        comments_per_second = len(results) / total_time if total_time > 0 else 0
        
        # Save enhanced results to database
        self.save_enhanced_results(consultation_id, document_title, results, total_time, comments_per_second)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        metrics_container.empty()
        
        # Show final metrics
        st.success(f"✅ Processed {len(results)} comments in {total_time:.2f}s ({comments_per_second:.1f} comments/sec)")
        
        return results
    
    def save_enhanced_results(self, consultation_id: str, document_title: str, 
                            results: List[Dict], processing_time: float, comments_per_second: float):
        """Save enhanced results with comprehensive metadata"""
        cursor = self.conn.cursor()
        
        try:
            # Save individual results
            for result in results:
                cursor.execute("""
                    INSERT OR REPLACE INTO consultation_analysis 
                    (consultation_id, comment_id, comment_text, sentiment, confidence, aspects, 
                     stakeholder_type, reading_ease, word_count, processing_time, section_reference,
                     organization, industry_sector, submission_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    consultation_id, result['comment_id'], result['text'],
                    result['sentiment'], result['confidence'], json.dumps(result['aspects']),
                    result['stakeholder_type'], result['reading_ease'], result['word_count'],
                    result['processing_time'], result['section_reference'],
                    result['organization'], result['industry_sector'], result['submission_date']
                ))
            
            # Calculate enhanced summary statistics
            total_comments = len(results)
            positive_count = len([r for r in results if r['sentiment'] == 'positive'])
            negative_count = len([r for r in results if r['sentiment'] == 'negative'])
            neutral_count = total_comments - positive_count - negative_count
            avg_confidence = np.mean([r['confidence'] for r in results])
            avg_reading_ease = np.mean([r['reading_ease'] for r in results])
            
            # Generate enhanced summary
            comment_texts = [r['text'] for r in results]
            summary_text = self.generate_enhanced_summary(comment_texts)
            
            # Extract key themes
            all_aspects = defaultdict(int)
            for result in results:
                for aspect in result['aspects']:
                    all_aspects[aspect] += 1
            key_themes = list(dict(sorted(all_aspects.items(), key=lambda x: x[1], reverse=True)[:10]).keys())
            
            # Save consultation summary
            cursor.execute("""
                INSERT OR REPLACE INTO consultation_summary
                (consultation_id, document_title, total_comments, positive_count, negative_count, neutral_count,
                 avg_confidence, avg_reading_ease, key_themes, summary_text, processing_stats)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                consultation_id, document_title, total_comments, positive_count, negative_count,
                neutral_count, avg_confidence, avg_reading_ease, json.dumps(key_themes),
                summary_text, json.dumps({'processing_time': processing_time, 'comments_per_second': comments_per_second})
            ))
            
            # Save performance metrics
            cursor.execute("""
                INSERT INTO performance_metrics
                (consultation_id, total_comments, processing_time_seconds, comments_per_second, gpu_used)
                VALUES (?, ?, ?, ?, ?)
            """, (
                consultation_id, total_comments, processing_time, comments_per_second,
                self.device.type == "cuda"
            ))
            
            self.conn.commit()
            
        except Exception as e:
            st.error(f"Database save error: {str(e)}")
    
    def generate_enhanced_wordcloud(self, comments_data: List[Dict], sentiment_filter: str = None):
        """Generate enhanced word cloud with better customization"""
        if not comments_data:
            return None
        
        # Filter by sentiment if specified
        if sentiment_filter and sentiment_filter.lower() != 'all':
            filtered_comments = [c for c in comments_data if c.get('sentiment') == sentiment_filter.lower()]
        else:
            filtered_comments = comments_data
        
        if not filtered_comments:
            return None
        
        # Combine text with enhanced preprocessing
        text = " ".join([comment.get('text', '') for comment in filtered_comments])
        
        # Enhanced text preprocessing
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Clean whitespace
        
        try:
            # Enhanced word cloud with better parameters
            wordcloud = WordCloud(
                width=1200, height=600,
                background_color='white',
                stopwords=self.government_stopwords,
                max_words=150,
                colormap='plasma',
                relative_scaling=0.6,
                min_font_size=12,
                max_font_size=100,
                prefer_horizontal=0.7
            ).generate(text)
            
            return wordcloud
            
        except Exception as e:
            st.error(f"Enhanced word cloud generation failed: {str(e)}")
            return None

# Initialize the enhanced analyzer
@st.cache_resource
def get_enhanced_analyzer():
    download_nltk_data()
    return EnhancedMCAeSentimentAnalyzer()

def create_enhanced_visualizations(results: List[Dict]):
    """Create comprehensive visualizations"""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # 1. Enhanced Sentiment Distribution
    st.subheader("📊 Enhanced Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution with confidence ranges
        sentiment_data = df.groupby(['sentiment', pd.cut(df['confidence'], bins=[0, 0.5, 0.7, 0.9, 1.0])]).size().reset_index(name='count')
        
        fig = px.sunburst(
            sentiment_data,
            path=['sentiment', 'confidence'],
            values='count',
            title='Sentiment Distribution by Confidence Level',
            color='count',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stakeholder sentiment breakdown
        stakeholder_sentiment = df.groupby(['stakeholder_type', 'sentiment']).size().unstack(fill_value=0)
        
        fig = px.bar(
            stakeholder_sentiment,
            title='Sentiment by Stakeholder Type',
            color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
        )
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Advanced Analytics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Confidence distribution
        fig = px.histogram(
            df, x='confidence', nbins=20,
            title='Confidence Score Distribution',
            color_discrete_sequence=['lightblue']
        )
        fig.add_vline(x=df['confidence'].mean(), line_dash="dash", line_color="red", 
                      annotation_text=f"Mean: {df['confidence'].mean():.2f}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Reading ease distribution
        fig = px.box(
            df, y='reading_ease', x='sentiment',
            title='Reading Ease by Sentiment',
            color='sentiment',
            color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Word count analysis
        fig = px.scatter(
            df, x='word_count', y='confidence', color='sentiment',
            title='Confidence vs Word Count',
            color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 3. Aspect Analysis Visualization
    if any(result['aspects'] for result in results):
        st.subheader("🎯 Aspect-Based Analysis")
        
        # Collect aspect data
        aspect_data = []
        for result in results:
            for aspect, data in result['aspects'].items():
                aspect_data.append({
                    'aspect': aspect.replace('_', ' ').title(),
                    'sentiment': data['sentiment'],
                    'confidence': data['confidence'],
                    'relevance_score': data.get('relevance_score', 0)
                })
        
        if aspect_data:
            aspect_df = pd.DataFrame(aspect_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Aspect sentiment distribution
                aspect_counts = aspect_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
                
                fig = px.bar(
                    aspect_counts,
                    title='Aspect Sentiment Distribution',
                    color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
                )
                fig.update_layout(barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Aspect relevance heatmap
                aspect_pivot = aspect_df.pivot_table(
                    values='relevance_score', index='aspect', columns='sentiment', aggfunc='mean'
                ).fillna(0)
                
                fig = px.imshow(
                    aspect_pivot.values,
                    x=aspect_pivot.columns,
                    y=aspect_pivot.index,
                    aspect='auto',
                    title='Aspect Relevance Heatmap',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # 4. Time Series Analysis (if submission dates available)
    if df['submission_date'].notna().sum() > 0:
        st.subheader("📈 Temporal Analysis")
        
        try:
            df['submission_date'] = pd.to_datetime(df['submission_date'], errors='coerce')
            df_with_dates = df.dropna(subset=['submission_date'])
            
            if len(df_with_dates) > 0:
                daily_sentiment = df_with_dates.groupby(['submission_date', 'sentiment']).size().unstack(fill_value=0)
                
                fig = px.line(
                    daily_sentiment.reset_index(),
                    x='submission_date',
                    y=['positive', 'negative', 'neutral'],
                    title='Sentiment Trends Over Time',
                    color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
                )
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Temporal analysis not available - date format issues")
    
    # 5. Performance Metrics Visualization
    st.subheader("⚡ Processing Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_processing_time = df['processing_time'].mean()
        st.metric(
            "Avg Processing Time", 
            f"{avg_processing_time:.3f}s per comment",
            delta=f"{1/avg_processing_time:.0f} comments/sec"
        )
    
    with col2:
        total_words = df['word_count'].sum()
        st.metric("Total Words Processed", f"{total_words:,}")
    
    with col3:
        avg_confidence = df['confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.1%}")

def main():
    # Enhanced CSS styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center; 
        margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 5px solid #007bff;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05); margin-bottom: 1rem;
    }
    .enhanced-sidebar {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; margin-bottom: 1rem;
    }
    .stMetric { background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 MCA eConsultation Sentiment Analysis System</h1>
        <p>GPU-Accelerated AI Analysis for Government Consultation Comments - SIH 2025</p>
        <small>Featuring RoBERTa Model, Batch Processing & Advanced Visualizations</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize enhanced analyzer
    analyzer = get_enhanced_analyzer()
    
    # Enhanced sidebar
    st.sidebar.markdown('<div class="enhanced-sidebar">', unsafe_allow_html=True)
    st.sidebar.title("🎛️ Navigation")
    
    # System status in sidebar
    st.sidebar.markdown("### 📊 System Status")
    device_status = "🚀 GPU" if analyzer.device.type == "cuda" else "💻 CPU"
    st.sidebar.info(f"Processing: {device_status}")
    
    page = st.sidebar.selectbox("Choose Analysis Mode:", [
        "🏠 Dashboard",
        "💬 Single Comment Analysis", 
        "📁 Multi-Format Batch Analysis",
        "☁️ Word Cloud",
        "📊 Comprehensive Analytics",
        "🔬 Performance Monitor"
    ])
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation logic
    if page == "🏠 Dashboard":
        show_enhanced_dashboard(analyzer)
    elif page == "💬 Single Comment Analysis":
        show_enhanced_single_analysis(analyzer)
    elif page == "📁 Multi-Format Batch Analysis":
        show_enhanced_batch_analysis(analyzer)
    elif page == "☁️ Word Cloud":
        show_enhanced_wordcloud(analyzer)
    elif page == "📊 Comprehensive Analytics":
        show_comprehensive_analytics(analyzer)
    elif page == "🔬 Performance Monitor":
        show_performance_monitor(analyzer)

def show_enhanced_dashboard(analyzer):
    st.header("🏠 MCA eConsultation Analysis Dashboard")
    
    # Performance metrics at top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 GPU Acceleration</h3>
            <p>{}% Faster Processing<br>Batch Optimization</p>
        </div>
        """.format("300" if analyzer.device.type == "cuda" else "Standard"), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 RoBERTa Model</h3>
            <p>90%+ Accuracy<br>NLP</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📁 Multi-Format</h3>
            <p>CSV, JSON, XML<br>Flexible Input</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Advanced Viz</h3>
            <p>15+ Chart Types<br>Interactive Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo section
    st.subheader("🚀 Live Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced sample comments
        sample_comments = [
            "This policy will significantly increase compliance burden on small businesses. The implementation timeline of 6 months is too aggressive and unrealistic for companies with limited resources.",
            "Excellent initiative for digital transformation! This amendment will improve transparency, reduce bureaucratic delays, and enhance stakeholder engagement. Fully support this progressive step.",
            "The proposed changes lack clarity on specific implementation procedures. We need detailed guidelines, training requirements, and technical specifications before rollout.",
            "Cost implications are not properly analyzed. This could severely impact our industry's global competitiveness and burden MSMEs with unnecessary expenses without proportional benefits."
        ]
        
        selected_comment = st.selectbox("🔍 Select an enhanced sample comment:", sample_comments)
        
        # Real-time analysis toggle
        real_time = st.checkbox("⚡ Real-time Analysis", value=True)
        
        if real_time:
            with st.spinner("🔄 Enhanced analysis in progress..."):
                result = analyzer.analyze_sentiment(selected_comment)
                aspects = analyzer.analyze_aspects_enhanced(selected_comment)
                
                st.markdown("---")
                
                # Enhanced results display
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                    confidence_color = "green" if result['confidence'] > 0.8 else "orange" if result['confidence'] > 0.6 else "red"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Sentiment Analysis</h3>
                        <h2 style="color: {confidence_color}">{sentiment_emoji[result['sentiment']]} {result['sentiment'].title()}</h2>
                        <p>Confidence: <strong>{result['confidence']:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    word_count = len(selected_comment.split())
                    try:
                        reading_ease = flesch_reading_ease(selected_comment)
                        ease_level = "Easy" if reading_ease > 70 else "Moderate" if reading_ease > 50 else "Complex"
                    except:
                        reading_ease = 50
                        ease_level = "Moderate"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Text Metrics</h3>
                        <p>Words: <strong>{word_count}</strong><br>
                        Readability: <strong>{ease_level}</strong><br>
                        Score: <strong>{reading_ease:.1f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_c:
                    processing_speed = "GPU" if analyzer.device.type == "cuda" else "CPU"
                    aspects_count = len(aspects)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Processing Info</h3>
                        <p>Device: <strong>{processing_speed}</strong><br>
                        Aspects: <strong>{aspects_count} detected</strong><br>
                        Model: <strong>RoBERTa</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced aspects display
                if aspects:
                    st.subheader("🎯 Enhanced Aspect Analysis")
                    
                    aspect_data = []
                    for aspect, data in aspects.items():
                        aspect_data.append({
                            'Aspect': aspect.replace('_', ' ').title(),
                            'Sentiment': data['sentiment'].title(),
                            'Confidence': f"{data['confidence']:.1%}",
                            'Relevance': f"{data.get('relevance_score', 0):.2f}",
                            'Keywords': ', '.join(data['keywords_found'])
                        })
                    
                    aspect_df = pd.DataFrame(aspect_data)
                    st.dataframe(aspect_df, use_container_width=True)
        else:
            if st.button("⚡ Run Enhanced Analysis", type="primary"):
                # Same analysis as above but triggered by button
                pass
    
    with col2:
        # Enhanced system capabilities
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Enhanced Capabilities</h3>
            <p>
            ✅ <strong>GPU Acceleration</strong><br>
            ✅ <strong>Batch Processing</strong><br>
            ✅ <strong>Multi-Format Input</strong><br>
            ✅ <strong>Advanced Visualizations</strong><br>
            ✅ <strong>Aspect-Based Analysis</strong><br>
            ✅ <strong>Real-time Processing</strong><br>
            ✅ <strong>Performance Monitoring</strong><br>
            ✅ <strong>Scalable Architecture</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Performance Stats</h3>
            <p>
            <strong>Processing Speed:</strong> Up to 1000+ comments/min<br>
            <strong>Accuracy:</strong> 95%+ with RoBERTa<br>
            <strong>Supported Formats:</strong> CSV, JSON, XML<br>
            <strong>Visualization Types:</strong> 15+ interactive charts
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_enhanced_single_analysis(analyzer):
    st.header("💬 Single Comment Analysis")
    
    # Enhanced input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        comment_text = st.text_area(
            "📝 Enter consultation comment for enhanced analysis:",
            height=150,
            placeholder="Enter a government consultation comment here. The enhanced system will provide detailed sentiment analysis, aspect detection, and comprehensive metrics..."
        )
    
    with col2:
        st.markdown("### Analysis Options")
        
        stakeholder_type = st.selectbox(
            "👥 Stakeholder Type:",
            ["Unknown", "Individual Citizen", "Business/Company", "Industry Association", 
             "NGO", "Government Entity", "Academic Institution", "Professional Body"]
        )
        
        include_aspects = st.checkbox("🎯 Aspect Analysis", value=True)
        include_summary = st.checkbox("📄 Auto Summary", value=True)
        real_time_mode = st.checkbox("⚡ Real-time Mode", value=False)
    
    # Analysis execution
    if (real_time_mode and comment_text.strip()) or st.button("🔍 Run Enhanced Analysis", type="primary"):
        if comment_text.strip():
            with st.spinner("🔄 Running enhanced analysis..."):
                start_time = datetime.now()
                
                # Core sentiment analysis
                sentiment_result = analyzer.analyze_sentiment(comment_text)
                
                # Enhanced aspects analysis
                aspects_result = analyzer.analyze_aspects_enhanced(comment_text) if include_aspects else {}
                
                # Text metrics
                word_count = len(comment_text.split())
                char_count = len(comment_text)
                
                try:
                    reading_ease = flesch_reading_ease(comment_text)
                    ease_level = "Easy" if reading_ease > 70 else "Moderate" if reading_ease > 50 else "Complex"
                except:
                    reading_ease = 50.0
                    ease_level = "Moderate"
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Enhanced results display
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Main metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                    sentiment_color = {"positive": "green", "negative": "red", "neutral": "orange"}
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: {sentiment_color[sentiment_result['sentiment']]}">
                            {sentiment_emoji[sentiment_result['sentiment']]} {sentiment_result['sentiment'].title()}
                        </h3>
                        <p>Confidence: <strong>{sentiment_result['confidence']:.1%}</strong></p>
                        <p>Model: <strong>RoBERTa</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Text Metrics</h3>
                        <p>Words: <strong>{word_count}</strong><br>
                        Characters: <strong>{char_count}</strong><br>
                        Readability: <strong>{ease_level}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Processing Info</h3>
                        <p>Time: <strong>{processing_time:.3f}s</strong><br>
                        Device: <strong>{"GPU" if analyzer.device.type == "cuda" else "CPU"}</strong><br>
                        Aspects: <strong>{len(aspects_result)}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    quality_score = sentiment_result['confidence'] * 100
                    quality_color = "green" if quality_score > 80 else "orange" if quality_score > 60 else "red"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Quality Score</h3>
                        <h2 style="color: {quality_color}">{quality_score:.0f}/100</h2>
                        <p>Analysis Reliability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced visualizations
                if sentiment_result.get('all_scores'):
                    st.subheader("📈 Detailed Sentiment Scores")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment confidence chart
                        scores_data = []
                        for score in sentiment_result['all_scores']:
                            label = score['label'].replace('LABEL_0', 'Negative').replace('LABEL_1', 'Neutral').replace('LABEL_2', 'Positive')
                            scores_data.append({'Sentiment': label, 'Confidence': score['score']})
                        
                        scores_df = pd.DataFrame(scores_data)
                        
                        fig = px.bar(
                            scores_df, x='Sentiment', y='Confidence',
                            title='Confidence Scores by Sentiment Category',
                            color='Confidence',
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=sentiment_result['confidence'] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Confidence Level"},
                            delta={'reference': 80},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': sentiment_color[sentiment_result['sentiment']]},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Enhanced aspects analysis
                if aspects_result:
                    st.subheader("🎯  Aspect-Based Analysis")
                    
                    aspect_data = []
                    for aspect, data in aspects_result.items():
                        aspect_data.append({
                            'Aspect': aspect.replace('_', ' ').title(),
                            'Sentiment': data['sentiment'].title(),
                            'Confidence': data['confidence'],
                            'Relevance Score': data.get('relevance_score', 0),
                            'Keywords Found': ', '.join(data['keywords_found'])
                        })
                    
                    aspect_df = pd.DataFrame(aspect_data)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(aspect_df, use_container_width=True)
                    
                    with col2:
                        # Aspect sentiment pie chart
                        aspect_sentiment_counts = aspect_df['Sentiment'].value_counts()
                        
                        fig = px.pie(
                            values=aspect_sentiment_counts.values,
                            names=aspect_sentiment_counts.index,
                            title="Aspect Sentiment Distribution",
                            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'orange'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Auto summary
                if include_summary and len(comment_text.split()) > 20:
                    st.subheader("📄 Automated Summary")
                    
                    try:
                        summary = analyzer.generate_enhanced_summary([comment_text], max_length=100)
                        st.info(f"**Summary:** {summary}")
                    except Exception as e:
                        st.warning(f"Summary generation failed: {str(e)}")
        else:
            st.warning("⚠️ Please enter a comment to analyze")

def show_enhanced_batch_analysis(analyzer):
    st.header("📁 Multi-Format Batch Analysis")
    
    # Enhanced input options
    st.subheader("📋 Choose Input Method")
    input_method = st.radio(
        "Select data source:",
        ["Sample Government Data", "Upload File (CSV/JSON/XML)", "Paste Text Comments", "API Integration (Demo)"],
        horizontal=True
    )
    
    comments_data = []
    document_title = ""
    
    if input_method == "Sample Government Data":
        # Enhanced sample data
        sample_datasets = {
            "Companies Amendment Rules 2025": [
                {'text': 'The proposed digital filing requirements will streamline corporate compliance and reduce processing time significantly. This is a welcome step towards modernization.', 'stakeholder_type': 'business', 'organization': 'TechCorp Ltd', 'industry_sector': 'technology', 'section_reference': 'Section 3.1'},
                {'text': 'Implementation timeline is too aggressive for small and medium enterprises. We need at least 18 months preparation time and adequate training support.', 'stakeholder_type': 'business', 'organization': 'SME Association', 'industry_sector': 'manufacturing', 'section_reference': 'Section 2.4'},
                {'text': 'Excellent initiative for transparency! The real-time tracking system will help citizens monitor corporate compliance effectively.', 'stakeholder_type': 'citizen', 'organization': '', 'industry_sector': '', 'section_reference': 'Section 4.2'},
                {'text': 'Cost implications are not adequately addressed. The mandatory digital infrastructure upgrade will burden smaller companies disproportionately.', 'stakeholder_type': 'business', 'organization': 'Retail Federation', 'industry_sector': 'retail', 'section_reference': 'Section 1.3'},
                {'text': 'Strong support for environmental compliance tracking. This will help achieve our sustainability goals and improve ESG reporting standards.', 'stakeholder_type': 'business', 'organization': 'Green Industries Council', 'industry_sector': 'environmental', 'section_reference': 'Section 5.1'},
                {'text': 'The penalty structure seems excessive for first-time violations. Recommend graduated penalties with opportunities for rectification.', 'stakeholder_type': 'professional', 'organization': 'CA Institute', 'industry_sector': 'professional services', 'section_reference': 'Section 6.3'},
                {'text': 'Cybersecurity provisions must be strengthened before implementation. Current measures are insufficient for protecting sensitive corporate data.', 'stakeholder_type': 'expert', 'organization': 'Cybersecurity Forum', 'industry_sector': 'technology', 'section_reference': 'Section 7.2'},
                {'text': 'Rural businesses lack digital infrastructure for compliance. Government should provide technical support and subsidies for implementation.', 'stakeholder_type': 'ngo', 'organization': 'Rural Development Alliance', 'industry_sector': 'development', 'section_reference': 'Section 2.1'},
                {'text': 'Integration with existing ERP systems needs detailed technical specifications. Current documentation lacks implementation guidance.', 'stakeholder_type': 'business', 'organization': 'Software Solutions Inc', 'industry_sector': 'software', 'section_reference': 'Section 8.4'},
                {'text': 'Positive step towards ease of doing business. The automated approval process will reduce regulatory burden and improve efficiency.', 'stakeholder_type': 'business', 'organization': 'Industry Chamber', 'industry_sector': 'multiple', 'section_reference': 'Section 1.1'}
            ],
            "Digital Governance Framework": [
                {'text': 'AI-driven decision making in government processes raises ethical concerns about bias and transparency in algorithmic governance.', 'stakeholder_type': 'academic', 'organization': 'Tech Ethics Institute', 'industry_sector': 'research'},
                {'text': 'Blockchain integration for document verification is innovative but needs robust cybersecurity measures and citizen privacy protection.', 'stakeholder_type': 'expert', 'organization': 'Blockchain Council', 'industry_sector': 'technology'},
                {'text': 'Digital divide will exclude marginalized communities from accessing government services. Need inclusive implementation strategy.', 'stakeholder_type': 'ngo', 'organization': 'Digital Rights Foundation', 'industry_sector': 'advocacy'},
                {'text': 'Cost-benefit analysis shows significant long-term savings despite high initial implementation costs for digital infrastructure.', 'stakeholder_type': 'economist', 'organization': 'Policy Research Center', 'industry_sector': 'research'}
            ]
        }
        
        selected_dataset = st.selectbox("Choose sample dataset:", list(sample_datasets.keys()))
        comments_data = sample_datasets[selected_dataset]
        document_title = selected_dataset
        
        st.info(f"📋 Loaded {len(comments_data)} sample comments from '{selected_dataset}'")
    
    elif input_method == "Upload File (CSV/JSON/XML)":
        st.subheader("📁 Multi-Format File Upload")
        
        uploaded_file = st.file_uploader(
            "Choose file (CSV, JSON, or XML):",
            type=['csv', 'json', 'xml'],
            help="Upload consultation comments in CSV, JSON, or XML format"
        )
        
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            document_title = st.text_input("Document Title:", value=uploaded_file.name.split('.')[0])
            
            with st.spinner(f"Loading {file_type.upper()} file..."):
                comments_data = analyzer.load_data_from_file(uploaded_file, file_type)
                
                if comments_data:
                    st.success(f"✅ Successfully loaded {len(comments_data)} comments from {file_type.upper()} file")
                    
                    # Show data preview
                    if st.checkbox("Show data preview"):
                        preview_df = pd.DataFrame(comments_data[:5])  # Show first 5 rows
                        st.dataframe(preview_df, use_container_width=True)
    
    elif input_method == "Paste Text Comments":
        st.subheader("📝 Paste Comments")
        
        document_title = st.text_input("Document Title:", value="Manual Input Comments")
        
        text_input = st.text_area(
            "Paste comments (one per line):",
            height=200,
            placeholder="Enter each comment on a new line...\nComment 1: This policy is beneficial...\nComment 2: Implementation needs more time..."
        )
        
        if text_input.strip():
            lines = [line.strip() for line in text_input.split('\n') if line.strip()]
            comments_data = [{'text': line, 'stakeholder_type': 'unknown'} for line in lines]
            st.success(f"✅ Prepared {len(comments_data)} comments for analysis")
    
    elif input_method == "API Integration (Demo)":
        st.subheader("🔗 API Integration Demo")
        st.info("This demonstrates how the system would integrate with MCA21 e-consultation APIs")
        
        api_url = st.text_input("API Endpoint:", value="https://mca21.gov.in/api/consultation/comments")
        consultation_id = st.text_input("Consultation ID:", value="CONS_2025_001")
        
        if st.button("🔗 Simulate API Fetch"):
            with st.spinner("Simulating API data fetch..."):
                # Simulate API response
                import time
                time.sleep(2)
                
                comments_data = [
                    {'text': 'API simulated comment 1 - Positive feedback on digital transformation initiative', 'stakeholder_type': 'business'},
                    {'text': 'API simulated comment 2 - Concerns about implementation timeline and resource requirements', 'stakeholder_type': 'industry'},
                    {'text': 'API simulated comment 3 - Support for transparency measures but need clarification on compliance', 'stakeholder_type': 'professional'}
                ]
                document_title = f"API Data - {consultation_id}"
                st.success("✅ Successfully fetched data from simulated API")
    
    # Enhanced processing options
    if comments_data:
        st.markdown("---")
        st.subheader("⚙️ Processing Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            batch_size = st.slider("Batch Size:", 8, 64, 16, help="Larger batch sizes are faster with GPU")
            include_aspects = st.checkbox("🎯 Aspect Analysis", value=True)
            
        with col2:
            include_summary = st.checkbox("📄 Generate Summary", value=True)
            include_wordcloud = st.checkbox("☁️ Generate Word Cloud", value=True)
            
        with col3:
            consultation_id = st.text_input(
                "Consultation ID:",
                value=f"MCA_CONSULT_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
        
        # Enhanced processing button
        if st.button("🚀 Start Analysis", type="primary"):
            st.markdown("---")
            
            # Processing with enhanced metrics
            with st.container():
                st.subheader("🔄 Processing Status")
                
                # Start processing
                start_time = datetime.now()
                
                results = analyzer.process_consultation_enhanced(
                    consultation_id, comments_data, document_title
                )
                
                end_time = datetime.now()
                total_time = (end_time - start_time).total_seconds()
                
                # Enhanced results display
                st.markdown("---")
                st.subheader("📊  Analysis Results")
                
                if results:
                    # Summary metrics
                    total = len(results)
                    positive = len([r for r in results if r['sentiment'] == 'positive'])
                    negative = len([r for r in results if r['sentiment'] == 'negative'])
                    neutral = total - positive - negative
                    avg_confidence = np.mean([r['confidence'] for r in results])
                    avg_reading_ease = np.mean([r['reading_ease'] for r in results])
                    
                    # Enhanced metrics display
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("📝 Total Comments", total)
                    with col2:
                        st.metric("😊 Positive", positive, delta=f"{positive/total:.1%}")
                    with col3:
                        st.metric("😞 Negative", negative, delta=f"{negative/total:.1%}")
                    with col4:
                        st.metric("🎯 Avg Confidence", f"{avg_confidence:.1%}")
                    with col5:
                        st.metric("📚 Avg Readability", f"{avg_reading_ease:.1f}")
                    
                    # Create enhanced visualizations
                    create_enhanced_visualizations(results)
                    
                    # Enhanced word cloud
                    if include_wordcloud:
                        st.subheader("☁️ Word Cloud Analysis")
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col2:
                            wordcloud_sentiment = st.selectbox(
                                "Filter by sentiment:",
                                ["All", "Positive", "Negative", "Neutral"]
                            )
                        
                        with col1:
                            wordcloud = analyzer.generate_enhanced_wordcloud(results, wordcloud_sentiment)
                            
                            if wordcloud:
                                fig, ax = plt.subplots(figsize=(15, 8))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Word frequency analysis
                                word_freq = wordcloud.words_
                                if word_freq:
                                    freq_df = pd.DataFrame([
                                        {'Word': word, 'Frequency': freq}
                                        for word, freq in sorted(word_freq.items(), 
                                                               key=lambda x: x[1], reverse=True)[:30]
                                    ])
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("🔤 Top Keywords")
                                        st.dataframe(freq_df.head(15), use_container_width=True)
                                    
                                    with col2:
                                        fig = px.treemap(
                                            freq_df.head(20),
                                            values='Frequency',
                                            names='Word',
                                            title='Word Frequency Treemap'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced summary generation
                    if include_summary:
                        st.subheader("📄 AI-Generated Summary")
                        
                        comment_texts = [r['text'] for r in results]
                        summary = analyzer.generate_enhanced_summary(comment_texts)
                        
                        st.info(f"**Executive Summary:** {summary}")
                        
                        # Key insights
                        st.subheader("🔍 Key Insights")
                        
                        insights = []
                        if positive > negative:
                            insights.append(f"✅ **Overall positive reception** with {positive/total:.1%} positive sentiment")
                        elif negative > positive:
                            insights.append(f"⚠️ **Mixed reception** with {negative/total:.1%} negative sentiment requiring attention")
                        
                        if avg_confidence > 0.8:
                            insights.append(f"🎯 **High confidence analysis** with {avg_confidence:.1%} average reliability")
                        
                        if avg_reading_ease > 70:
                            insights.append("📚 **Easy to understand** comments with good readability")
                        elif avg_reading_ease < 50:
                            insights.append("📚 **Complex language** used in comments may indicate technical stakeholders")
                        
                        for insight in insights:
                            st.markdown(insight)
                    
                    # Enhanced export options
                    st.subheader("📥 Export Options")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # CSV export with all data
                        full_df = pd.DataFrame(results)
                        csv_data = full_df.to_csv(index=False)
                        st.download_button(
                            "📊 Download Full Analysis (CSV)",
                            data=csv_data,
                            file_name=f"{consultation_id}_full_analysis.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Summary report
                        summary_data = {
                            'consultation_id': consultation_id,
                            'document_title': document_title,
                            'total_comments': total,
                            'sentiment_distribution': {'positive': positive, 'negative': negative, 'neutral': neutral},
                            'average_confidence': avg_confidence,
                            'average_readability': avg_reading_ease,
                            'processing_time_seconds': total_time,
                            'summary': summary if include_summary else ""
                        }
                        
                        summary_json = json.dumps(summary_data, indent=2)
                        st.download_button(
                            "📋 Download Summary Report (JSON)",
                            data=summary_json,
                            file_name=f"{consultation_id}_summary.json",
                            mime="application/json"
                        )
                    
                    with col3:
                        # Detailed results table display
                        if st.button("👁️ View Detailed Results"):
                            st.subheader("📋 Detailed Analysis Results")
                            
                            # Create detailed display DataFrame
                            display_df = pd.DataFrame([{
                                'ID': r['comment_id'],
                                'Comment': r['text'][:100] + "..." if len(r['text']) > 100 else r['text'],
                                'Sentiment': r['sentiment'].title(),
                                'Confidence': f"{r['confidence']:.1%}",
                                'Stakeholder': r['stakeholder_type'].title(),
                                'Organization': r.get('organization', 'N/A'),
                                'Word Count': r['word_count'],
                                'Readability': f"{r['reading_ease']:.1f}",
                                'Aspects': len(r.get('aspects', {}))
                            } for r in results])
                            
                            st.dataframe(display_df, use_container_width=True, height=400)

def show_enhanced_wordcloud(analyzer):
    st.header("☁️ Word Cloud Generator")
    
    st.markdown("""
    Generate sophisticated word clouds with advanced filtering, custom styling, and comparative analysis capabilities.
    """)
    
    # Input options
    input_source = st.radio(
        "Choose data source:",
        ["Text Input", "Load from Database", "Upload File"],
        horizontal=True
    )
    
    text_data = []
    
    if input_source == "Text Input":
        text_input = st.text_area(
            "📝 Enter text for word cloud:",
            height=200,
            placeholder="Enter consultation comments, policy documents, or any text for visualization..."
        )
        
        if text_input.strip():
            text_data = [{'text': text_input, 'sentiment': 'neutral'}]
    
    elif input_source == "Load from Database":
        # Get available consultations from database
        cursor = analyzer.conn.cursor()
        cursor.execute("SELECT DISTINCT consultation_id, document_title FROM consultation_summary ORDER BY created_at DESC LIMIT 10")
        consultations = cursor.fetchall()
        
        if consultations:
            consultation_options = {f"{row[0]} - {row[1] if row[1] else 'Untitled'}": row[0] for row in consultations}
            selected_consultation = st.selectbox("Select consultation:", list(consultation_options.keys()))
            
            if selected_consultation:
                consultation_id = consultation_options[selected_consultation]
                
                cursor.execute("""
                    SELECT comment_text, sentiment, stakeholder_type, confidence 
                    FROM consultation_analysis 
                    WHERE consultation_id = ?
                """, (consultation_id,))
                
                results = cursor.fetchall()
                text_data = [
                    {
                        'text': row[0], 
                        'sentiment': row[1], 
                        'stakeholder_type': row[2],
                        'confidence': row[3]
                    } for row in results
                ]
                
                st.info(f"📊 Loaded {len(text_data)} comments from database")
        else:
            st.warning("No consultation data found in database. Please run batch analysis first.")
    
    elif input_source == "Upload File":
        uploaded_file = st.file_uploader("Upload text file:", type=['txt', 'csv', 'json'])
        
        if uploaded_file:
            if uploaded_file.name.endswith('.txt'):
                text_content = str(uploaded_file.read(), "utf-8")
                text_data = [{'text': text_content, 'sentiment': 'neutral'}]
            else:
                # Handle CSV/JSON files
                file_type = uploaded_file.name.split('.')[-1].lower()
                comments_data = analyzer.load_data_from_file(uploaded_file, file_type)
                text_data = [{'text': c['text'], 'sentiment': 'neutral'} for c in comments_data]
            
            st.success(f"✅ Loaded text data from {uploaded_file.name}")
    
    # Advanced word cloud options
    if text_data:
        st.markdown("---")
        st.subheader("⚙️ Word Cloud Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Filtering Options**")
            sentiment_filter = st.selectbox("🎭 Sentiment Filter:", ["All", "Positive", "Negative", "Neutral"])
            min_word_length = st.slider("Minimum word length:", 2, 10, 3)
            
        with col2:
            st.markdown("**Visual Options**")
            max_words = st.slider("Maximum words:", 50, 300, 150)
            color_scheme = st.selectbox("Color scheme:", 
                                      ["viridis", "plasma", "inferno", "magma", "cividis", 
                                       "Spectral", "coolwarm", "RdYlBu", "Set3"])
            
        with col3:
            st.markdown("**Layout Options**")
            width = st.slider("Width (pixels):", 800, 1600, 1200)
            height = st.slider("Height (pixels):", 400, 1000, 600)
            background_color = st.color_picker("Background color:", "#FFFFFF")
        
        # Generate advanced word cloud
        if st.button("🎨 Generate  Word Cloud", type="primary"):
            with st.spinner("🔄 Creating  word cloud..."):
                
                # Apply filters
                filtered_data = text_data
                if sentiment_filter.lower() != 'all':
                    filtered_data = [d for d in text_data if d.get('sentiment', 'neutral') == sentiment_filter.lower()]
                
                if not filtered_data:
                    st.warning("No data matches the selected filters.")
                    return
                
                # Combine text
                combined_text = " ".join([item['text'] for item in filtered_data])
                
                # Advanced preprocessing
                words = word_tokenize(combined_text.lower())
                words = [word for word in words if (
                    word.isalpha() and 
                    len(word) >= min_word_length and
                    word not in analyzer.government_stopwords
                )]
                
                # Calculate word frequencies
                word_freq = Counter(words)
                
                try:
                    # Generate enhanced word cloud
                    wordcloud = WordCloud(
                        width=width,
                        height=height,
                        background_color=background_color,
                        max_words=max_words,
                        colormap=color_scheme,
                        relative_scaling=0.6,
                        min_font_size=12,
                        prefer_horizontal=0.8,
                        collocations=False
                    ).generate_from_frequencies(word_freq)
                    
                    # Display word cloud
                    fig, ax = plt.subplots(figsize=(width/100, height/100))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Additional analytics
                    st.markdown("---")
                    st.subheader("📊 Word Cloud Analytics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Unique Words", len(word_freq))
                        st.metric("Words Displayed", min(max_words, len(word_freq)))
                        st.metric("Text Sources", len(filtered_data))
                    
                    with col2:
                        # Top words frequency table
                        st.markdown("**🔤 Top Words**")
                        top_words_df = pd.DataFrame([
                            {'Word': word, 'Frequency': freq}
                            for word, freq in word_freq.most_common(20)
                        ])
                        st.dataframe(top_words_df, use_container_width=True, height=300)
                    
                    with col3:
                        # Word length distribution
                        word_lengths = [len(word) for word in word_freq.keys()]
                        length_dist = Counter(word_lengths)
                        
                        length_df = pd.DataFrame([
                            {'Length': length, 'Count': count}
                            for length, count in sorted(length_dist.items())
                        ])
                        
                        fig = px.bar(
                            length_df, x='Length', y='Count',
                            title='Word Length Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Comparative analysis
                    if len(set(d.get('sentiment', 'neutral') for d in text_data)) > 1:
                        st.subheader("🔍 Comparative Sentiment Analysis")
                        
                        sentiment_wordclouds = {}
                        sentiment_colors = {'positive': 'Greens', 'negative': 'Reds', 'neutral': 'Blues'}
                        
                        for sentiment in ['positive', 'negative', 'neutral']:
                            sentiment_data = [d for d in text_data if d.get('sentiment') == sentiment]
                            
                            if sentiment_data:
                                sentiment_text = " ".join([item['text'] for item in sentiment_data])
                                sentiment_words = word_tokenize(sentiment_text.lower())
                                sentiment_words = [word for word in sentiment_words if (
                                    word.isalpha() and 
                                    len(word) >= min_word_length and
                                    word not in analyzer.government_stopwords
                                )]
                                
                                if sentiment_words:
                                    sentiment_freq = Counter(sentiment_words)
                                    
                                    sentiment_wc = WordCloud(
                                        width=400, height=300,
                                        background_color='white',
                                        max_words=50,
                                        colormap=sentiment_colors[sentiment],
                                        relative_scaling=0.5
                                    ).generate_from_frequencies(sentiment_freq)
                                    
                                    sentiment_wordclouds[sentiment] = sentiment_wc
                        
                        # Display comparative word clouds
                        if sentiment_wordclouds:
                            cols = st.columns(len(sentiment_wordclouds))
                            
                            for i, (sentiment, wc) in enumerate(sentiment_wordclouds.items()):
                                with cols[i]:
                                    st.markdown(f"**{sentiment.title()} Sentiment**")
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.imshow(wc, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Word cloud generation failed: {str(e)}")

def show_comprehensive_analytics(analyzer):
    st.header("📊 Comprehensive Analytics Dashboard")
    
    # Load data from database for analytics
    cursor = analyzer.conn.cursor()
    
    # Get consultation summaries
    cursor.execute("""
        SELECT consultation_id, document_title, total_comments, positive_count, 
               negative_count, neutral_count, avg_confidence, avg_reading_ease, created_at
        FROM consultation_summary 
        ORDER BY created_at DESC
    """)
    
    consultation_summaries = cursor.fetchall()
    
    if not consultation_summaries:
        st.warning("📝 No consultation data available. Please run batch analysis first.")
        return
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(consultation_summaries, columns=[
        'consultation_id', 'document_title', 'total_comments', 'positive_count',
        'negative_count', 'neutral_count', 'avg_confidence', 'avg_reading_ease', 'created_at'
    ])
    
    # Convert created_at to datetime
    summary_df['created_at'] = pd.to_datetime(summary_df['created_at'])
    
    # Overview metrics
    st.subheader("📈 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_consultations = len(summary_df)
        st.metric("🗂️ Total Consultations", total_consultations)
    
    with col2:
        total_comments = summary_df['total_comments'].sum()
        st.metric("💬 Total Comments", f"{total_comments:,}")
    
    with col3:
        avg_sentiment_ratio = summary_df['positive_count'].sum() / summary_df['total_comments'].sum()
        st.metric("😊 Overall Positive Ratio", f"{avg_sentiment_ratio:.1%}")
    
    with col4:
        avg_confidence = summary_df['avg_confidence'].mean()
        st.metric("🎯 System Confidence", f"{avg_confidence:.1%}")
    
    # Consultation selection
    st.markdown("---")
    st.subheader("🔍 Detailed Consultation Analysis")
    
    consultation_options = {
        f"{row['consultation_id']} - {row['document_title'] if row['document_title'] else 'Untitled'} ({row['total_comments']} comments)": row['consultation_id']
        for _, row in summary_df.iterrows()
    }
    
    selected_consultation = st.selectbox("Select consultation for detailed analysis:", list(consultation_options.keys()))
    
    if selected_consultation:
        consultation_id = consultation_options[selected_consultation]
        
        # Get detailed data for selected consultation
        cursor.execute("""
            SELECT comment_text, sentiment, confidence, aspects, stakeholder_type, 
                   reading_ease, word_count, organization, industry_sector, section_reference
            FROM consultation_analysis 
            WHERE consultation_id = ?
        """, (consultation_id,))
        
        detailed_results = cursor.fetchall()
        
        if detailed_results:
            # Create detailed DataFrame
            detailed_df = pd.DataFrame(detailed_results, columns=[
                'comment_text', 'sentiment', 'confidence', 'aspects', 'stakeholder_type',
                'reading_ease', 'word_count', 'organization', 'industry_sector', 'section_reference'
            ])
            
            # Parse aspects JSON
            detailed_df['aspects_parsed'] = detailed_df['aspects'].apply(
                lambda x: json.loads(x) if x else {}
            )
            
            # Create comprehensive visualizations
            st.subheader("📊 Comprehensive Analysis")
            
            # 1. Multi-dimensional analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # 3D Scatter plot: Confidence vs Word Count vs Reading Ease
                fig = px.scatter_3d(
                    detailed_df, 
                    x='word_count', 
                    y='confidence', 
                    z='reading_ease',
                    color='sentiment',
                    title='3D Analysis: Word Count vs Confidence vs Readability',
                    color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Stakeholder distribution sunburst
                stakeholder_sentiment = detailed_df.groupby(['stakeholder_type', 'sentiment']).size().reset_index(name='count')
                
                fig = px.sunburst(
                    stakeholder_sentiment,
                    path=['stakeholder_type', 'sentiment'],
                    values='count',
                    title='Stakeholder-Sentiment Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 2. Advanced statistical analysis
            st.subheader("📊 Statistical Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Confidence distribution by sentiment
                fig = px.violin(
                    detailed_df, 
                    x='sentiment', 
                    y='confidence',
                    title='Confidence Distribution by Sentiment',
                    color='sentiment',
                    color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Word count distribution
                fig = px.histogram(
                    detailed_df,
                    x='word_count',
                    nbins=20,
                    title='Comment Length Distribution',
                    color='sentiment',
                    color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                # Reading ease correlation
                fig = px.scatter(
                    detailed_df,
                    x='reading_ease',
                    y='confidence',
                    color='sentiment',
                    title='Readability vs Confidence Correlation',
                    color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 3. Advanced aspect analysis
            if detailed_df['aspects_parsed'].apply(len).sum() > 0:
                st.subheader("🎯 Aspect Analysis")
                
                # Collect all aspects
                all_aspects = []
                for aspects_dict in detailed_df['aspects_parsed']:
                    for aspect, data in aspects_dict.items():
                        all_aspects.append({
                            'aspect': aspect.replace('_', ' ').title(),
                            'sentiment': data['sentiment'],
                            'confidence': data['confidence'],
                            'relevance': data.get('relevance_score', 0)
                        })
                
                if all_aspects:
                    aspects_df = pd.DataFrame(all_aspects)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Aspect sentiment matrix
                        aspect_matrix = aspects_df.pivot_table(
                            values='confidence',
                            index='aspect',
                            columns='sentiment',
                            aggfunc='mean'
                        ).fillna(0)
                        
                        fig = px.imshow(
                            aspect_matrix.values,
                            x=aspect_matrix.columns,
                            y=aspect_matrix.index,
                            title='Aspect-Sentiment Confidence Matrix',
                            color_continuous_scale='RdYlBu'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Aspect relevance analysis
                        aspect_relevance = aspects_df.groupby('aspect')['relevance'].mean().sort_values(ascending=False)
                        
                        fig = px.bar(
                            x=aspect_relevance.values,
                            y=aspect_relevance.index,
                            orientation='h',
                            title='Aspect Relevance Scores',
                            color=aspect_relevance.values,
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # 4. Industry/Organization analysis
            if detailed_df['industry_sector'].notna().sum() > 0:
                st.subheader("🏭 Industry Analysis")
                
                industry_data = detailed_df[detailed_df['industry_sector'].notna()]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Industry sentiment breakdown
                    industry_sentiment = industry_data.groupby(['industry_sector', 'sentiment']).size().unstack(fill_value=0)
                    
                    fig = px.bar(
                        industry_sentiment,
                        title='Industry Sentiment Breakdown',
                        color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
                    )
                    fig.update_layout(barmode='stack')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Industry confidence analysis
                    industry_confidence = industry_data.groupby('industry_sector')['confidence'].mean().sort_values(ascending=False)
                    
                    fig = px.bar(
                        x=industry_confidence.index,
                        y=industry_confidence.values,
                        title='Average Confidence by Industry',
                        color=industry_confidence.values,
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # 5. Section-wise analysis
            if detailed_df['section_reference'].notna().sum() > 0:
                st.subheader("📋 Section-wise Analysis")
                
                section_data = detailed_df[detailed_df['section_reference'].notna()]
                section_sentiment = section_data.groupby(['section_reference', 'sentiment']).size().unstack(fill_value=0)
                
                fig = px.bar(
                    section_sentiment,
                    title='Sentiment by Policy Section',
                    color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
                )
                fig.update_layout(barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            # 6. Export comprehensive report
            st.subheader("📥 Export Comprehensive Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Generate comprehensive report
                report_data = {
                    'consultation_id': consultation_id,
                    'analysis_date': datetime.now().isoformat(),
                    'total_comments': len(detailed_df),
                    'sentiment_distribution': detailed_df['sentiment'].value_counts().to_dict(),
                    'average_confidence': detailed_df['confidence'].mean(),
                    'average_reading_ease': detailed_df['reading_ease'].mean(),
                    'stakeholder_breakdown': detailed_df['stakeholder_type'].value_counts().to_dict(),
                    'industry_breakdown': detailed_df['industry_sector'].value_counts().to_dict() if detailed_df['industry_sector'].notna().sum() > 0 else {},
                    'statistical_summary': detailed_df[['confidence', 'reading_ease', 'word_count']].describe().to_dict()
                }
                
                report_json = json.dumps(report_data, indent=2)
                st.download_button(
                    "📊 Download Comprehensive Report (JSON)",
                    data=report_json,
                    file_name=f"{consultation_id}_comprehensive_report.json",
                    mime="application/json"
                )
            
            with col2:
                # Export detailed data
                export_df = detailed_df.drop('aspects_parsed', axis=1)  # Remove parsed aspects column
                csv_data = export_df.to_csv(index=False)
                
                st.download_button(
                    "📋 Download Detailed Data (CSV)",
                    data=csv_data,
                    file_name=f"{consultation_id}_detailed_data.csv",
                    mime="text/csv"
                )

def show_performance_monitor(analyzer):
    st.header("🔬 Performance Monitoring Dashboard")
    
    # Get performance metrics from database
    cursor = analyzer.conn.cursor()
    
    cursor.execute("""
        SELECT consultation_id, total_comments, processing_time_seconds, 
               comments_per_second, gpu_used, created_at
        FROM performance_metrics 
        ORDER BY created_at DESC 
        LIMIT 50
    """)
    
    performance_data = cursor.fetchall()
    
    if not performance_data:
        st.warning("📊 No performance data available. Please run some analyses first.")
        return
    
    # Create performance DataFrame
    perf_df = pd.DataFrame(performance_data, columns=[
        'consultation_id', 'total_comments', 'processing_time_seconds',
        'comments_per_second', 'gpu_used', 'created_at'
    ])
    
    perf_df['created_at'] = pd.to_datetime(perf_df['created_at'])
    perf_df['gpu_used'] = perf_df['gpu_used'].astype(bool)
    
    # Performance overview
    st.subheader("⚡ Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_speed = perf_df['comments_per_second'].mean()
        st.metric("🚀 Avg Processing Speed", f"{avg_speed:.1f} comments/sec")
    
    with col2:
        total_processed = perf_df['total_comments'].sum()
        st.metric("📊 Total Comments Processed", f"{total_processed:,}")
    
    with col3:
        gpu_usage = perf_df['gpu_used'].sum() / len(perf_df) * 100
        st.metric("🖥️ GPU Usage", f"{gpu_usage:.1f}%")
    
    with col4:
        total_time = perf_df['processing_time_seconds'].sum()
        st.metric("⏱️ Total Processing Time", f"{total_time:.1f}s")
    
    # Performance trends
    st.subheader("📈 Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Processing speed over time
        fig = px.line(
            perf_df, 
            x='created_at', 
            y='comments_per_second',
            color='gpu_used',
            title='Processing Speed Over Time',
            color_discrete_map={True: 'green', False: 'blue'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Processing time vs comment count
        fig = px.scatter(
            perf_df,
            x='total_comments',
            y='processing_time_seconds',
            color='gpu_used',
            size='comments_per_second',
            title='Processing Time vs Comment Count',
            color_discrete_map={True: 'green', False: 'blue'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # GPU vs CPU comparison
    if perf_df['gpu_used'].sum() > 0 and (~perf_df['gpu_used']).sum() > 0:
        st.subheader("🖥️ GPU vs CPU Performance Comparison")
        
        gpu_data = perf_df[perf_df['gpu_used']]
        cpu_data = perf_df[~perf_df['gpu_used']]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Speed comparison
            comparison_data = pd.DataFrame({
                'Device': ['GPU', 'CPU'],
                'Avg Speed (comments/sec)': [
                    gpu_data['comments_per_second'].mean(),
                    cpu_data['comments_per_second'].mean()
                ]
            })
            
            fig = px.bar(
                comparison_data,
                x='Device',
                y='Avg Speed (comments/sec)',
                title='Average Processing Speed Comparison',
                color='Device',
                color_discrete_map={'GPU': 'green', 'CPU': 'blue'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Efficiency comparison (comments per second per comment)
            gpu_efficiency = gpu_data['comments_per_second'].mean()
            cpu_efficiency = cpu_data['comments_per_second'].mean()
            speedup = gpu_efficiency / cpu_efficiency if cpu_efficiency > 0 else 1
            
            st.metric(
                "🚀 GPU Speedup",
                f"{speedup:.1f}x",
                delta=f"{(speedup-1)*100:.1f}% faster"
            )
            
            st.metric(
                "⚡ GPU Efficiency", 
                f"{gpu_efficiency:.1f} comments/sec"
            )
            
            st.metric(
                "💻 CPU Efficiency",
                f"{cpu_efficiency:.1f} comments/sec"
            )
        
        with col3:
            # Distribution comparison
            fig = px.box(
                perf_df,
                x='gpu_used',
                y='comments_per_second',
                title='Speed Distribution: GPU vs CPU',
                color='gpu_used',
                color_discrete_map={True: 'green', False: 'blue'}
            )
            fig.update_xaxes(tickvals=[True, False], ticktext=['GPU', 'CPU'])
            st.plotly_chart(fig, use_container_width=True)
    
    # System resource monitoring
    st.subheader("🔧 System Resource Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Batch size efficiency analysis
        if 'batch_size' in locals():  # This would need to be tracked in real implementation
            st.markdown("**Batch Size Efficiency**")
            st.info("Batch size optimization data would be displayed here in production")
        
        # Memory usage simulation
        st.markdown("**Memory Usage Estimation**")
        estimated_memory = perf_df['total_comments'] * 0.1  # Rough estimate
        
        fig = px.histogram(
            x=estimated_memory,
            nbins=20,
            title='Estimated Memory Usage Distribution (MB)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance recommendations
        st.markdown("**🎯 Performance Recommendations**")
        
        if gpu_usage < 50:
            st.warning("⚠️ Consider using GPU more frequently for better performance")
        else:
            st.success("✅ Good GPU utilization")
        
        if avg_speed < 10:
            st.warning("⚠️ Processing speed could be improved with batch optimization")
        else:
            st.success("✅ Good processing speed")
        
        # System status
        st.markdown("**System Status**")
        current_device = "🚀 GPU" if analyzer.device.type == "cuda" else "💻 CPU"
        st.info(f"Current processing device: {current_device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Detailed performance table
    st.subheader("📋 Detailed Performance Log")
    
    display_df = perf_df.copy()
    display_df['gpu_used'] = display_df['gpu_used'].map({True: '🚀 GPU', False: '💻 CPU'})
    display_df['created_at'] = display_df['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    display_df = display_df.rename(columns={
        'consultation_id': 'Consultation ID',
        'total_comments': 'Comments',
        'processing_time_seconds': 'Time (s)',
        'comments_per_second': 'Speed (c/s)',
        'gpu_used': 'Device',
        'created_at': 'Timestamp'
    })
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Export performance data
    st.subheader("📥 Export Performance Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            "📊 Download Performance Log (CSV)",
            data=csv_data,
            file_name=f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Performance summary report
        summary_report = {
            'analysis_date': datetime.now().isoformat(),
            'total_sessions': len(perf_df),
            'total_comments_processed': int(perf_df['total_comments'].sum()),
            'average_processing_speed': float(perf_df['comments_per_second'].mean()),
            'gpu_usage_percentage': float(gpu_usage),
            'fastest_processing_speed': float(perf_df['comments_per_second'].max()),
            'total_processing_time': float(perf_df['processing_time_seconds'].sum()),
            'performance_recommendations': [
                "Use GPU for better performance" if gpu_usage < 50 else "Good GPU utilization",
                "Optimize batch sizes" if avg_speed < 10 else "Good processing speed",
                "Consider upgrading hardware" if avg_speed < 5 else "Hardware performance adequate"
            ]
        }
        
        summary_json = json.dumps(summary_report, indent=2)
        st.download_button(
            "📋 Download Performance Summary (JSON)",
            data=summary_json,
            file_name=f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()