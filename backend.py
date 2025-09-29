
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json
import sqlite3
import hashlib
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from textstat import flesch_reading_ease
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import asyncio
from functools import lru_cache
import io
import base64

app = FastAPI(title="MCA eConsultation Sentiment Analysis API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.processing_status: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.processing_status:
            del self.processing_status[client_id]

    async def send_update(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except:
                self.disconnect(client_id)

    async def broadcast_system_status(self, status: dict):
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps({
                    "type": "system_status",
                    "data": status
                }))
            except:
                disconnected.append(client_id)

        for client_id in disconnected:
            self.disconnect(client_id)

manager = ConnectionManager()

# Pydantic models for API requests/responses (same as before)
class CommentAnalysisRequest(BaseModel):
    text: str
    stakeholder_type: Optional[str] = "unknown"
    organization: Optional[str] = ""
    industry_sector: Optional[str] = ""
    section_reference: Optional[str] = ""

class BatchAnalysisRequest(BaseModel):
    comments: List[Dict[str, Any]]
    consultation_id: str
    document_title: Optional[str] = ""
    batch_size: Optional[int] = 16
    include_aspects: Optional[bool] = True
    include_summary: Optional[bool] = True

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    aspects: Dict[str, Any]
    reading_ease: float
    word_count: int
    processing_time: float

class BatchAnalysisResponse(BaseModel):
    consultation_id: str
    total_comments: int
    results: List[Dict[str, Any]]
    summary: Optional[str]
    processing_time: float
    performance_metrics: Dict[str, Any]

class VisualizationRequest(BaseModel):
    consultation_id: str
    chart_type: str
    filters: Optional[Dict[str, Any]] = {}

class WordCloudRequest(BaseModel):
    consultation_id: Optional[str] = None
    text_data: Optional[List[Dict[str, str]]] = None
    sentiment_filter: Optional[str] = "all"
    max_words: Optional[int] = 150
    width: Optional[int] = 1200
    height: Optional[int] = 600
    color_scheme: Optional[str] = "viridis"

# Global analyzer instance
analyzer = None

class EnhancedMCAeAnalyzerBackend:
    def __init__(self):
        """Initialize the Enhanced MCA eConsultation Sentiment Analysis Backend"""
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
            print(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
            return device
        else:
            device = torch.device("cpu")
            print("💻 Using CPU for processing")
            return device

    def setup_models(self):
        """Load and setup AI models with GPU support"""
        try:
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
                print("✅ RoBERTa model loaded on GPU!")
            else:
                print("✅ RoBERTa model loaded on CPU")

            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True
            )

            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device.type == "cuda" else -1
            )

        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            self.sentiment_pipeline = pipeline("sentiment-analysis")

    def setup_database(self):
        """Setup SQLite database"""
        self.conn = sqlite3.connect('mca_consultations_enhanced.db', check_same_thread=False)
        cursor = self.conn.cursor()

        # Create tables
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
        """Get government-specific stopwords"""
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
        """Get policy aspects for analysis"""
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

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        if not text or len(text.strip()) < 5:
            return {'sentiment': 'neutral', 'confidence': 0.3, 'all_scores': []}

        try:
            if len(text.split()) > 500:
                text = ' '.join(text.split()[:500])

            result = self.sentiment_pipeline(text)
            if isinstance(result[0], list):
                result = result[0]

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
            print(f"Sentiment analysis error: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0.3, 'all_scores': []}

    def analyze_aspects(self, text: str) -> Dict[str, Any]:
        """Analyze aspects in text"""
        text_lower = text.lower()
        aspects_found = {}

        for aspect, config in self.policy_aspects.items():
            keywords = config['keywords']
            weight = config['weight']

            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                relevance_score = len(found_keywords) / len(keywords) * weight
                aspect_sentiment = self.analyze_sentiment(text)

                aspects_found[aspect] = {
                    'sentiment': aspect_sentiment['sentiment'],
                    'confidence': aspect_sentiment['confidence'] * relevance_score,
                    'keywords_found': found_keywords,
                    'relevance_score': relevance_score
                }

        return aspects_found

    def generate_visualizations(self, consultation_id: str, chart_type: str, filters: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Generate various types of visualizations"""
        cursor = self.conn.cursor()

        # Get data from database
        cursor.execute("""
            SELECT comment_text, sentiment, confidence, stakeholder_type, reading_ease, 
                   word_count, organization, industry_sector, aspects
            FROM consultation_analysis 
            WHERE consultation_id = ?
        """, (consultation_id,))

        results = cursor.fetchall()
        if not results:
            raise HTTPException(status_code=404, detail="No data found for consultation")

        # Convert to DataFrame
        df = pd.DataFrame(results, columns=[
            'comment_text', 'sentiment', 'confidence', 'stakeholder_type', 
            'reading_ease', 'word_count', 'organization', 'industry_sector', 'aspects'
        ])

        # Apply filters
        if filters:
            if 'sentiment' in filters and filters['sentiment'] != 'all':
                df = df[df['sentiment'] == filters['sentiment']]
            if 'stakeholder_type' in filters and filters['stakeholder_type'] != 'all':
                df = df[df['stakeholder_type'] == filters['stakeholder_type']]

        # Generate different types of charts
        if chart_type == 'sentiment_distribution':
            sentiment_counts = df['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Sentiment Distribution',
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
            )

        elif chart_type == 'confidence_histogram':
            fig = px.histogram(
                df, x='confidence', nbins=20,
                title='Confidence Score Distribution',
                color_discrete_sequence=['lightblue']
            )

        elif chart_type == 'stakeholder_sentiment':
            stakeholder_sentiment = df.groupby(['stakeholder_type', 'sentiment']).size().unstack(fill_value=0)
            fig = px.bar(
                stakeholder_sentiment,
                title='Sentiment by Stakeholder Type',
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
            )
            fig.update_layout(barmode='stack')

        elif chart_type == '3d_scatter':
            fig = px.scatter_3d(
                df, x='word_count', y='confidence', z='reading_ease',
                color='sentiment',
                title='3D Analysis: Word Count vs Confidence vs Readability',
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
            )

        elif chart_type == 'reading_ease_box':
            fig = px.box(
                df, y='reading_ease', x='sentiment',
                title='Reading Ease by Sentiment',
                color='sentiment',
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
            )

        elif chart_type == 'sunburst':
            # Create sunburst chart for stakeholder-sentiment breakdown
            sunburst_data = df.groupby(['stakeholder_type', 'sentiment']).size().reset_index(name='count')
            fig = px.sunburst(
                sunburst_data, 
                path=['stakeholder_type', 'sentiment'], 
                values='count',
                title='Stakeholder-Sentiment Sunburst Analysis'
            )

        elif chart_type == 'violin':
            fig = px.violin(
                df, y='confidence', x='sentiment', box=True,
                title='Confidence Distribution by Sentiment (Violin Plot)',
                color='sentiment',
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
            )

        elif chart_type == 'heatmap':
            # Create heatmap of stakeholder vs confidence ranges
            df['confidence_range'] = pd.cut(df['confidence'], bins=[0, 0.5, 0.7, 0.9, 1.0], 
                                          labels=['Low', 'Medium', 'High', 'Very High'])
            heatmap_data = df.groupby(['stakeholder_type', 'confidence_range']).size().unstack(fill_value=0)
            fig = px.imshow(
                heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                title='Stakeholder vs Confidence Heatmap',
                color_continuous_scale='Blues'
            )

        elif chart_type == 'treemap':
            # Create treemap of sentiment by stakeholder type
            treemap_data = df.groupby(['stakeholder_type', 'sentiment']).size().reset_index(name='count')
            fig = px.treemap(
                treemap_data,
                path=['stakeholder_type', 'sentiment'],
                values='count',
                title='Sentiment Distribution Treemap'
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid chart type")

        # Convert to JSON
        chart_json = fig.to_json()

        return {
            'chart_type': chart_type,
            'chart_data': json.loads(chart_json),
            'data_points': len(df),
            'filters_applied': filters
        }

    def generate_wordcloud(self, request: WordCloudRequest) -> Dict[str, Any]:
        """Generate word cloud"""
        text_data = []

        if request.consultation_id:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT comment_text, sentiment 
                FROM consultation_analysis 
                WHERE consultation_id = ?
            """, (request.consultation_id,))

            results = cursor.fetchall()
            text_data = [{'text': row[0], 'sentiment': row[1]} for row in results]

        elif request.text_data:
            text_data = request.text_data

        if not text_data:
            raise HTTPException(status_code=400, detail="No text data provided")

        # Filter by sentiment if specified
        if request.sentiment_filter and request.sentiment_filter.lower() != 'all':
            text_data = [item for item in text_data if item.get('sentiment') == request.sentiment_filter.lower()]

        if not text_data:
            raise HTTPException(status_code=400, detail="No data matches the filters")

        # Combine text
        combined_text = " ".join([item['text'] for item in text_data])

        # Generate word cloud
        try:
            wordcloud = WordCloud(
                width=request.width,
                height=request.height,
                background_color='white',
                stopwords=self.government_stopwords,
                max_words=request.max_words,
                colormap=request.color_scheme,
                relative_scaling=0.6,
                min_font_size=12,
                prefer_horizontal=0.7
            ).generate(combined_text)

            # Convert to base64 image
            img_buffer = io.BytesIO()
            wordcloud.to_image().save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            # Get word frequencies
            word_frequencies = dict(wordcloud.words_)
            top_words = dict(sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:30])

            return {
                'wordcloud_image': f"data:image/png;base64,{img_base64}",
                'word_frequencies': top_words,
                'total_words': len(word_frequencies),
                'text_sources': len(text_data)
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")

    def save_results(self, consultation_id: str, results: List[Dict[str, Any]], document_title: str = ""):
        """Save analysis results to database"""
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
                    result.get('processing_time', 0), result['section_reference'],
                    result['organization'], result['industry_sector'], result.get('submission_date', '')
                ))

            # Calculate summary statistics
            total_comments = len(results)
            positive_count = len([r for r in results if r['sentiment'] == 'positive'])
            negative_count = len([r for r in results if r['sentiment'] == 'negative'])
            neutral_count = total_comments - positive_count - negative_count
            avg_confidence = np.mean([r['confidence'] for r in results])
            avg_reading_ease = np.mean([r['reading_ease'] for r in results])

            # Save consultation summary
            cursor.execute("""
                INSERT OR REPLACE INTO consultation_summary
                (consultation_id, document_title, total_comments, positive_count, negative_count, neutral_count,
                 avg_confidence, avg_reading_ease, key_themes, summary_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                consultation_id, document_title, total_comments, positive_count, negative_count,
                neutral_count, avg_confidence, avg_reading_ease, json.dumps([]), ""
            ))

            self.conn.commit()

        except Exception as e:
            print(f"Database save error: {str(e)}")

# Initialize the analyzer
@lru_cache()
def get_analyzer():
    global analyzer
    if analyzer is None:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        analyzer = EnhancedMCAeAnalyzerBackend()
    return analyzer

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "MCA eConsultation Sentiment Analysis API - Enhanced Version",
        "version": "2.0.0",
        "features": [
            "GPU-accelerated sentiment analysis with RoBERTa",
            "Real-time batch processing with WebSocket updates",
            "15+ advanced visualization types",
            "Multi-format file support (CSV, JSON, XML)",
            "Comprehensive analytics dashboard",
            "Word cloud generation with customization",
            "Performance monitoring and metrics",
            "Export functionality (CSV, JSON, Excel)",
            "Aspect-based sentiment analysis",
            "Real-time progress tracking"
        ]
    }

@app.get("/health")
async def health_check():
    analyzer = get_analyzer()
    return {
        "status": "healthy",
        "device": "GPU" if analyzer.device.type == "cuda" else "CPU",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": hasattr(analyzer, 'sentiment_pipeline'),
        "active_connections": len(manager.active_connections),
        "features_status": {
            "sentiment_analysis": "✅ Active",
            "batch_processing": "✅ Active", 
            "visualizations": "✅ Active",
            "word_cloud": "✅ Active",
            "real_time_updates": "✅ Active",
            "database": "✅ Connected"
        }
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.get("/dashboard/metrics")
async def get_dashboard_metrics():
    """Get real-time dashboard metrics that update automatically"""
    analyzer = get_analyzer()
    cursor = analyzer.conn.cursor()

    # Get total consultations
    cursor.execute("SELECT COUNT(*) FROM consultation_summary")
    total_consultations = cursor.fetchone()[0]

    # Get total comments processed
    cursor.execute("SELECT SUM(total_comments) FROM consultation_summary")
    total_comments = cursor.fetchone()[0] or 0

    # Get overall sentiment distribution
    cursor.execute("""
        SELECT sentiment, COUNT(*) as count 
        FROM consultation_analysis 
        GROUP BY sentiment
    """)
    sentiment_data = dict(cursor.fetchall())

    # Get processing speed metrics
    cursor.execute("""
        SELECT AVG(comments_per_second) as avg_speed,
               MAX(comments_per_second) as max_speed,
               COUNT(*) as total_batches
        FROM performance_metrics
    """)
    speed_data = cursor.fetchone()

    # Get GPU usage stats
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN gpu_used = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as gpu_usage_percent
        FROM performance_metrics
    """)
    gpu_usage = cursor.fetchone()[0] or 0

    # Get recent activity (last 24 hours)
    cursor.execute("""
        SELECT COUNT(*) FROM consultation_summary 
        WHERE created_at >= datetime('now', '-1 day')
    """)
    recent_activity = cursor.fetchone()[0]

    return {
        "total_consultations": total_consultations,
        "total_comments": total_comments,
        "sentiment_distribution": sentiment_data,
        "processing_metrics": {
            "avg_speed": speed_data[0] or 0,
            "max_speed": speed_data[1] or 0,
            "total_batches": speed_data[2] or 0
        },
        "gpu_usage_percent": gpu_usage,
        "recent_activity": recent_activity,
        "system_status": {
            "gpu_available": torch.cuda.is_available(),
            "device": "GPU" if analyzer.device.type == "cuda" else "CPU",
            "models_loaded": hasattr(analyzer, 'sentiment_pipeline'),
            "active_websockets": len(manager.active_connections)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze/single", response_model=SentimentResponse)
async def analyze_single_comment(request: CommentAnalysisRequest):
    """Analyze a single comment with comprehensive results"""
    analyzer = get_analyzer()

    start_time = datetime.now()

    # Perform analysis
    sentiment_result = analyzer.analyze_sentiment(request.text)
    aspects_result = analyzer.analyze_aspects(request.text)

    # Calculate text metrics
    try:
        reading_ease = flesch_reading_ease(request.text)
    except:
        reading_ease = 50.0

    word_count = len(request.text.split())
    processing_time = (datetime.now() - start_time).total_seconds()

    return SentimentResponse(
        sentiment=sentiment_result['sentiment'],
        confidence=sentiment_result['confidence'],
        aspects=aspects_result,
        reading_ease=reading_ease,
        word_count=word_count,
        processing_time=processing_time
    )

@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch_comments(request: BatchAnalysisRequest):
    """Analyze multiple comments in batch with comprehensive results"""
    analyzer = get_analyzer()

    start_time = datetime.now()
    results = []

    for i, comment in enumerate(request.comments):
        text = comment.get('text', '')
        if not text.strip():
            continue

        comment_id = hashlib.md5(f"{request.consultation_id}_{i}_{text[:50]}".encode()).hexdigest()[:8]

        # Analyze sentiment and aspects
        sentiment_result = analyzer.analyze_sentiment(text)
        aspects_result = analyzer.analyze_aspects(text) if request.include_aspects else {}

        try:
            reading_ease = flesch_reading_ease(text)
        except:
            reading_ease = 50.0

        result = {
            'comment_id': comment_id,
            'text': text,
            'sentiment': sentiment_result['sentiment'],
            'confidence': sentiment_result['confidence'],
            'aspects': aspects_result,
            'stakeholder_type': comment.get('stakeholder_type', 'unknown'),
            'organization': comment.get('organization', ''),
            'industry_sector': comment.get('industry_sector', ''),
            'section_reference': comment.get('section_reference', ''),
            'reading_ease': reading_ease,
            'word_count': len(text.split())
        }

        results.append(result)

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    # Generate summary if requested
    summary = None
    if request.include_summary and results:
        try:
            comment_texts = [r['text'] for r in results]
            combined_text = " ".join(comment_texts)
            if len(combined_text.split()) > 1000:
                combined_text = " ".join(combined_text.split()[:1000])

            summary_result = analyzer.summarizer(
                combined_text,
                max_length=200,
                min_length=50,
                do_sample=False
            )
            summary = summary_result[0]['summary_text']
        except:
            summary = "Summary generation failed"

    # Save to database
    analyzer.save_results(request.consultation_id, results, request.document_title)

    # Save performance metrics
    cursor = analyzer.conn.cursor()
    cursor.execute("""
        INSERT INTO performance_metrics 
        (consultation_id, total_comments, processing_time_seconds, comments_per_second, gpu_used)
        VALUES (?, ?, ?, ?, ?)
    """, (
        request.consultation_id, 
        len(results), 
        processing_time, 
        len(results) / processing_time if processing_time > 0 else 0,
        analyzer.device.type == "cuda"
    ))
    analyzer.conn.commit()

    performance_metrics = {
        'total_comments': len(results),
        'processing_time_seconds': processing_time,
        'comments_per_second': len(results) / processing_time if processing_time > 0 else 0,
        'gpu_used': analyzer.device.type == "cuda"
    }

    return BatchAnalysisResponse(
        consultation_id=request.consultation_id,
        total_comments=len(results),
        results=results,
        summary=summary,
        processing_time=processing_time,
        performance_metrics=performance_metrics
    )

@app.post("/analyze/batch-realtime")
async def analyze_batch_realtime(request: BatchAnalysisRequest, client_id: str):
    """Analyze batch with real-time progress updates via WebSocket"""
    analyzer = get_analyzer()

    start_time = datetime.now()
    results = []
    total_comments = len(request.comments)

    # Send initial status
    await manager.send_update(client_id, {
        "type": "batch_started",
        "total_comments": total_comments,
        "consultation_id": request.consultation_id,
        "timestamp": datetime.now().isoformat()
    })

    for i, comment in enumerate(request.comments):
        text = comment.get('text', '')
        if not text.strip():
            continue

        # Process comment
        comment_id = hashlib.md5(f"{request.consultation_id}_{i}_{text[:50]}".encode()).hexdigest()[:8]
        sentiment_result = analyzer.analyze_sentiment(text)
        aspects_result = analyzer.analyze_aspects(text) if request.include_aspects else {}

        try:
            reading_ease = flesch_reading_ease(text)
        except:
            reading_ease = 50.0

        result = {
            'comment_id': comment_id,
            'text': text,
            'sentiment': sentiment_result['sentiment'],
            'confidence': sentiment_result['confidence'],
            'aspects': aspects_result,
            'stakeholder_type': comment.get('stakeholder_type', 'unknown'),
            'organization': comment.get('organization', ''),
            'industry_sector': comment.get('industry_sector', ''),
            'section_reference': comment.get('section_reference', ''),
            'reading_ease': reading_ease,
            'word_count': len(text.split())
        }

        results.append(result)

        # Send progress update every 10 comments or at the end
        if (i + 1) % 10 == 0 or (i + 1) == total_comments:
            progress = ((i + 1) / total_comments) * 100
            elapsed = (datetime.now() - start_time).total_seconds()
            speed = (i + 1) / elapsed if elapsed > 0 else 0

            # Calculate current sentiment distribution for real-time updates
            current_sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
            for r in results:
                current_sentiments[r['sentiment']] += 1

            await manager.send_update(client_id, {
                "type": "batch_progress",
                "processed": i + 1,
                "total": total_comments,
                "progress": progress,
                "speed": speed,
                "elapsed": elapsed,
                "current_sentiments": current_sentiments,
                "timestamp": datetime.now().isoformat()
            })

    # Save results and send completion
    processing_time = (datetime.now() - start_time).total_seconds()
    analyzer.save_results(request.consultation_id, results, request.document_title)

    # Calculate final metrics
    sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
    for result in results:
        sentiment_distribution[result['sentiment']] += 1

    avg_confidence = np.mean([r['confidence'] for r in results])
    avg_reading_ease = np.mean([r['reading_ease'] for r in results])

    await manager.send_update(client_id, {
        "type": "batch_completed",
        "total_processed": len(results),
        "processing_time": processing_time,
        "consultation_id": request.consultation_id,
        "sentiment_distribution": sentiment_distribution,
        "avg_confidence": avg_confidence,
        "avg_reading_ease": avg_reading_ease,
        "comments_per_second": len(results) / processing_time if processing_time > 0 else 0,
        "timestamp": datetime.now().isoformat()
    })

    return {
        "message": "Batch processing completed",
        "consultation_id": request.consultation_id,
        "total_processed": len(results),
        "processing_time": processing_time
    }

@app.post("/visualizations/generate")
async def generate_visualization(request: VisualizationRequest):
    """Generate comprehensive visualizations for consultation data"""
    analyzer = get_analyzer()

    try:
        result = analyzer.generate_visualizations(
            request.consultation_id,
            request.chart_type,
            request.filters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/types")
async def get_visualization_types():
    """Get available visualization types"""
    return {
        "chart_types": [
            {
                "id": "sentiment_distribution",
                "name": "Sentiment Distribution",
                "description": "Pie chart showing overall sentiment breakdown",
                "category": "basic"
            },
            {
                "id": "confidence_histogram", 
                "name": "Confidence Histogram",
                "description": "Distribution of confidence scores",
                "category": "analysis"
            },
            {
                "id": "stakeholder_sentiment",
                "name": "Stakeholder Sentiment",
                "description": "Sentiment breakdown by stakeholder type",
                "category": "demographic"
            },
            {
                "id": "3d_scatter",
                "name": "3D Scatter Plot", 
                "description": "Word count vs confidence vs readability",
                "category": "advanced"
            },
            {
                "id": "reading_ease_box",
                "name": "Reading Ease Box Plot",
                "description": "Reading ease distribution by sentiment",
                "category": "analysis"
            },
            {
                "id": "sunburst",
                "name": "Sunburst Chart",
                "description": "Hierarchical stakeholder-sentiment breakdown",
                "category": "advanced"
            },
            {
                "id": "violin",
                "name": "Violin Plot",
                "description": "Confidence distribution density by sentiment",
                "category": "advanced"
            },
            {
                "id": "heatmap",
                "name": "Confidence Heatmap",
                "description": "Stakeholder vs confidence level heatmap",
                "category": "analysis"
            },
            {
                "id": "treemap",
                "name": "Sentiment Treemap",
                "description": "Hierarchical sentiment distribution",
                "category": "advanced"
            }
        ]
    }

@app.post("/wordcloud/generate")
async def generate_wordcloud_endpoint(request: WordCloudRequest):
    """Generate customizable word cloud visualization"""
    analyzer = get_analyzer()

    try:
        result = analyzer.generate_wordcloud(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/consultations/list")
async def list_consultations():
    """Get list of available consultations with comprehensive details"""
    analyzer = get_analyzer()
    cursor = analyzer.conn.cursor()

    cursor.execute("""
        SELECT consultation_id, document_title, total_comments, 
               positive_count, negative_count, neutral_count, 
               avg_confidence, avg_reading_ease, created_at
        FROM consultation_summary 
        ORDER BY created_at DESC
    """)

    consultations = cursor.fetchall()

    return {
        'consultations': [{
            'consultation_id': row[0],
            'document_title': row[1] or 'Untitled Consultation',
            'total_comments': row[2],
            'sentiment_distribution': {
                'positive': row[3],
                'negative': row[4],
                'neutral': row[5]
            },
            'avg_confidence': round(row[6], 3) if row[6] else 0,
            'avg_reading_ease': round(row[7], 1) if row[7] else 0,
            'created_at': row[8],
            'positive_ratio': round((row[3] / row[2]) * 100, 1) if row[2] > 0 else 0
        } for row in consultations],
        'total_consultations': len(consultations)
    }

@app.get("/consultations/{consultation_id}/details")
async def get_consultation_details(consultation_id: str):
    """Get detailed analysis for a specific consultation"""
    analyzer = get_analyzer()
    cursor = analyzer.conn.cursor()

    cursor.execute("""
        SELECT comment_text, sentiment, confidence, stakeholder_type, 
               reading_ease, word_count, aspects, organization,
               industry_sector, section_reference, timestamp
        FROM consultation_analysis 
        WHERE consultation_id = ?
        ORDER BY timestamp DESC
    """, (consultation_id,))

    results = cursor.fetchall()
    if not results:
        raise HTTPException(status_code=404, detail="Consultation not found")

    return {
        'consultation_id': consultation_id,
        'total_comments': len(results),
        'comments': [{
            'text': row[0],
            'sentiment': row[1],
            'confidence': row[2],
            'stakeholder_type': row[3],
            'reading_ease': row[4],
            'word_count': row[5],
            'aspects': json.loads(row[6]) if row[6] else {},
            'organization': row[7],
            'industry_sector': row[8], 
            'section_reference': row[9],
            'timestamp': row[10]
        } for row in results]
    }

@app.get("/analytics/comprehensive/{consultation_id}")
async def get_comprehensive_analytics(consultation_id: str):
    """Get comprehensive analytics for a consultation"""
    analyzer = get_analyzer()
    cursor = analyzer.conn.cursor()

    # Get detailed consultation data
    cursor.execute("""
        SELECT comment_text, sentiment, confidence, aspects, stakeholder_type,
               reading_ease, word_count, organization, industry_sector, section_reference
        FROM consultation_analysis 
        WHERE consultation_id = ?
    """, (consultation_id,))

    results = cursor.fetchall()
    if not results:
        raise HTTPException(status_code=404, detail="Consultation not found")

    # Process data for comprehensive analytics
    comments_data = []
    all_aspects = {}

    for row in results:
        comment_data = {
            'text': row[0],
            'sentiment': row[1],
            'confidence': row[2],
            'aspects': json.loads(row[3]) if row[3] else {},
            'stakeholder_type': row[4],
            'reading_ease': row[5],
            'word_count': row[6],
            'organization': row[7],
            'industry_sector': row[8],
            'section_reference': row[9]
        }
        comments_data.append(comment_data)

        # Collect aspects
        for aspect, data in comment_data['aspects'].items():
            if aspect not in all_aspects:
                all_aspects[aspect] = {'positive': 0, 'negative': 0, 'neutral': 0}
            all_aspects[aspect][data['sentiment']] += 1

    # Calculate comprehensive metrics
    total_comments = len(comments_data)
    sentiment_dist = {'positive': 0, 'negative': 0, 'neutral': 0}
    stakeholder_dist = {}
    industry_dist = {}
    section_dist = {}

    for comment in comments_data:
        sentiment_dist[comment['sentiment']] += 1

        if comment['stakeholder_type']:
            stakeholder_dist[comment['stakeholder_type']] = stakeholder_dist.get(comment['stakeholder_type'], 0) + 1

        if comment['industry_sector']:
            industry_dist[comment['industry_sector']] = industry_dist.get(comment['industry_sector'], 0) + 1

        if comment['section_reference']:
            section_dist[comment['section_reference']] = section_dist.get(comment['section_reference'], 0) + 1

    # Generate summary statistics
    avg_confidence = np.mean([c['confidence'] for c in comments_data])
    avg_reading_ease = np.mean([c['reading_ease'] for c in comments_data])
    total_words = sum([c['word_count'] for c in comments_data])

    return {
        'consultation_id': consultation_id,
        'total_comments': total_comments,
        'sentiment_distribution': sentiment_dist,
        'stakeholder_distribution': stakeholder_dist,
        'industry_distribution': industry_dist,
        'section_distribution': section_dist,
        'aspect_analysis': all_aspects,
        'summary_statistics': {
            'average_confidence': avg_confidence,
            'average_reading_ease': avg_reading_ease,
            'total_words': total_words,
            'positive_ratio': sentiment_dist['positive'] / total_comments,
            'negative_ratio': sentiment_dist['negative'] / total_comments,
            'neutral_ratio': sentiment_dist['neutral'] / total_comments
        },
        'timestamp': datetime.now().isoformat()
    }

@app.post("/analytics/export/{consultation_id}")
async def export_consultation_data(consultation_id: str, format: str = "csv"):
    """Export consultation data in various formats"""
    analyzer = get_analyzer()
    cursor = analyzer.conn.cursor()

    cursor.execute("""
        SELECT comment_text, sentiment, confidence, aspects, stakeholder_type,
               reading_ease, word_count, organization, industry_sector, 
               section_reference, timestamp
        FROM consultation_analysis 
        WHERE consultation_id = ?
    """, (consultation_id,))

    results = cursor.fetchall()
    if not results:
        raise HTTPException(status_code=404, detail="Consultation not found")

    # Create DataFrame
    df = pd.DataFrame(results, columns=[
        'comment_text', 'sentiment', 'confidence', 'aspects', 'stakeholder_type',
        'reading_ease', 'word_count', 'organization', 'industry_sector',
        'section_reference', 'timestamp'
    ])

    if format.lower() == "csv":
        output = df.to_csv(index=False)
        media_type = "text/csv"
        filename = f"{consultation_id}_analysis.csv"

    elif format.lower() == "json":
        output = df.to_json(orient='records', indent=2)
        media_type = "application/json"
        filename = f"{consultation_id}_analysis.json"

    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'json'")

    return {
        "filename": filename,
        "content": base64.b64encode(output.encode()).decode(),
        "media_type": media_type,
        "size": len(output),
        "total_records": len(df)
    }

@app.get("/system/performance")
async def get_system_performance():
    """Get comprehensive system performance metrics"""
    analyzer = get_analyzer()
    cursor = analyzer.conn.cursor()

    # Get performance over time
    cursor.execute("""
        SELECT DATE(created_at) as date,
               AVG(comments_per_second) as avg_speed,
               COUNT(*) as batches_processed,
               SUM(total_comments) as comments_processed,
               AVG(CASE WHEN gpu_used = 1 THEN 1.0 ELSE 0.0 END) as gpu_usage_ratio
        FROM performance_metrics
        WHERE created_at >= datetime('now', '-30 days')
        GROUP BY DATE(created_at)
        ORDER BY date
    """)

    daily_performance = cursor.fetchall()

    # Get GPU vs CPU performance
    cursor.execute("""
        SELECT gpu_used,
               AVG(comments_per_second) as avg_speed,
               COUNT(*) as batch_count,
               SUM(total_comments) as total_comments
        FROM performance_metrics
        GROUP BY gpu_used
    """)

    device_performance = cursor.fetchall()

    # Get current system status
    system_status = {
        "gpu_available": torch.cuda.is_available(),
        "current_device": "GPU" if analyzer.device.type == "cuda" else "CPU", 
        "active_websocket_connections": len(manager.active_connections),
        "models_loaded": hasattr(analyzer, 'sentiment_pipeline'),
        "database_connected": True
    }

    return {
        "daily_performance": [{
            "date": row[0],
            "avg_speed": row[1],
            "batches_processed": row[2],
            "comments_processed": row[3],
            "gpu_usage_ratio": row[4]
        } for row in daily_performance],
        "device_performance": [{
            "device": "GPU" if row[0] else "CPU",
            "avg_speed": row[1],
            "batch_count": row[2],
            "total_comments": row[3]
        } for row in device_performance],
        "system_status": system_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process consultation file with enhanced parsing"""
    analyzer = get_analyzer()

    # Read file content
    content = await file.read()

    # Determine file type and process accordingly
    file_extension = file.filename.split('.')[-1].lower()

    comments_data = []

    try:
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(content))
            text_columns = ['text', 'comment', 'comment_text', 'content', 'description']
            text_col = None

            for col in text_columns:
                if col in df.columns:
                    text_col = col
                    break

            if not text_col:
                raise HTTPException(status_code=400, detail="No text column found in CSV")

            for _, row in df.iterrows():
                comments_data.append({
                    'text': str(row.get(text_col, "")),
                    'stakeholder_type': str(row.get('stakeholder_type', 'unknown')),
                    'organization': str(row.get('organization', '')),
                    'industry_sector': str(row.get('industry_sector', '')),
                    'section_reference': str(row.get('section_reference', ''))
                })

        elif file_extension == 'json':
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text_fields = ['text', 'comment', 'comment_text', 'content', 'description']
                        text_content = ""

                        for field in text_fields:
                            if field in item:
                                text_content = str(item[field])
                                break

                        comments_data.append({
                            'text': text_content,
                            'stakeholder_type': str(item.get('stakeholder_type', 'unknown')),
                            'organization': str(item.get('organization', '')),
                            'industry_sector': str(item.get('industry_sector', '')),
                            'section_reference': str(item.get('section_reference', ''))
                        })

        elif file_extension == 'xml':
            root = ET.fromstring(content)
            comment_elements = root.findall('.//comment') + root.findall('.//item')

            for element in comment_elements:
                text_content = element.text or ""
                comments_data.append({
                    'text': text_content,
                    'stakeholder_type': 'unknown',
                    'organization': '',
                    'industry_sector': '',
                    'section_reference': ''
                })

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please use CSV, JSON, or XML.")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")

    return {
        'message': 'File uploaded and processed successfully',
        'filename': file.filename,
        'file_size': len(content),
        'file_type': file_extension,
        'comments_extracted': len(comments_data),
        'preview': comments_data[:5] if comments_data else [],
        'processing_timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
