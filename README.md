# MCA eConsultation Sentiment Analysis Backend API

A GPU-accelerated FastAPI backend for analyzing sentiment in government consultation comments using advanced AI models.
For now only run the enhanced_mca_sentiment_app.py
use command :
```bash
streamlit run enhanced_mca_sentiment_app.py
```
## Features

- **GPU-Accelerated Processing**: Utilizes RoBERTa model with GPU support for high-performance sentiment analysis
- **Batch Processing**: Efficiently processes multiple comments simultaneously
- **Multi-Format Support**: Accepts CSV, JSON, and XML file uploads
- **Aspect-Based Analysis**: Identifies policy aspects like economic impact, compliance, transparency, etc.
- **Word Cloud Generation**: Creates visual word clouds from comment data
- **Comprehensive Analytics**: Provides detailed sentiment analysis and statistics
- **Database Storage**: SQLite database for storing analysis results and performance metrics

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download required NLTK data (automatically handled on first run)

## Running the API

```bash
python backend.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Core Analysis

- `GET /` - API information
- `GET /health` - Health check
- `POST /analyze/sentiment` - Analyze single text sentiment
- `POST /analyze/batch` - Analyze multiple texts
- `POST /analyze/consultation` - Full consultation analysis

### File Upload

- `POST /upload` - Upload and analyze CSV/JSON/XML files

### Data Visualization

- `POST /wordcloud` - Generate word cloud
- `GET /analytics` - Get all consultations summary
- `GET /analytics/{consultation_id}` - Get specific consultation analytics

### Data Export

- `GET /export/{consultation_id}` - Export consultation data (JSON/CSV)

## Usage Examples

### Single Text Analysis

```python
import requests

response = requests.post("http://localhost:8000/analyze/sentiment",
    json={"text": "This policy will greatly benefit small businesses."}
)
print(response.json())
```

### Batch Analysis

```python
import requests

texts = [
    "Excellent initiative for digital transformation!",
    "Implementation timeline is too aggressive.",
    "Strong support for transparency measures."
]

response = requests.post("http://localhost:8000/analyze/batch",
    json={"texts": texts, "batch_size": 16}
)
print(response.json())
```

### File Upload Analysis

```python
import requests

files = {"file": open("consultation_comments.csv", "rb")}
data = {"consultation_id": "CONS_2025_001", "document_title": "Digital Governance Framework"}

response = requests.post("http://localhost:8000/upload", files=files, data=data)
print(response.json())
```

## Response Format

### Sentiment Analysis Response
```json
{
  "sentiment": "positive",
  "confidence": 0.9876,
  "all_scores": [
    {"label": "LABEL_2", "score": 0.9876},
    {"label": "LABEL_1", "score": 0.0102},
    {"label": "LABEL_0", "score": 0.0022}
  ]
}
```

### Consultation Analysis Response
```json
{
  "consultation_id": "CONS_2025_001",
  "total_comments": 150,
  "results": [...],
  "status": "completed"
}
```

## Performance

- **GPU Support**: Automatic detection and utilization of CUDA GPUs
- **Batch Processing**: Up to 1000+ comments per minute with GPU
- **Caching**: LRU cache for repeated sentiment analysis
- **Async Processing**: Non-blocking API endpoints

## Database

The API uses SQLite database (`mca_consultations_enhanced.db`) to store:
- Individual comment analysis results
- Consultation summaries
- Performance metrics
- Analysis metadata

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Transformers**: Hugging Face transformers for AI models
- **PyTorch**: Deep learning framework with GPU support
- **Pandas**: Data manipulation and analysis
- **Plotly**: Data visualization
- **NLTK**: Natural language processing tools
- **TextStat**: Readability analysis

## Development

The backend is built with:
- **FastAPI** for the web framework
- **Pydantic** for data validation
- **Uvicorn** as the ASGI server
- **Logging** for monitoring and debugging

## License

This project is part of the SIH 2025 initiative for MCA eConsultation analysis.
