# Investment Analysis Dashboard

A professional AI-powered investment analysis platform that generates comprehensive sector reports using real-time web scraping and advanced language models.

## Features

- **Real-time Data Scraping**: Automatically searches and scrapes relevant financial news and analysis
- **AI-Powered Analysis**: Uses Groq's DeepSeek model for detailed investment reports
- **Professional Reports**: Generates institutional-quality investment analysis with specific recommendations
- **Indian Market Focus**: Specialized for Indian equity markets and sectors
- **Rate Limiting**: Built-in protection against abuse
- **Responsive UI**: Modern, professional frontend interface

## Tech Stack

### Backend
- **FastAPI**: High-performance async web framework
- **Groq API**: Advanced language model for analysis
- **SerpAPI**: Web search functionality
- **aiohttp**: Async HTTP client for web scraping
- **BeautifulSoup4**: HTML parsing
- **Trafilatura**: Content extraction
- **Crawl4AI**: Optional advanced web crawler

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **Modern CSS**: Professional UI with animations
- **Responsive Design**: Works on all devices

## Project Structure

```
investment-analysis/
├── cd.py                 # FastAPI backend application
├── index.html           # Frontend interface
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
└── README.md           # This file
```

## Installation

### Prerequisites
- Python 3.8+
- Groq API Key
- SerpAPI Key

### Setup

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file**:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   SERPAPI_KEY=your_serpapi_key_here
   ```

4. **Get API Keys**:
   - **Groq API**: Sign up at [https://console.groq.com](https://console.groq.com)
   - **SerpAPI**: Sign up at [https://serpapi.com](https://serpapi.com)

## Local Development

1. **Start the server**:
   ```bash
   python cd.py
   ```

2. **Access the application**:
   - **Frontend**: http://localhost:8000/
   - **API Documentation**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health

## API Endpoints

### Main Endpoints
- `GET /` - Serves the main HTML interface
- `GET /analyze/{sector}` - Generate investment analysis for a sector
- `GET /health` - Health check and service status
- `GET /sectors` - List of recommended sectors

### Example API Usage
```bash
# Analyze Technology sector
curl http://localhost:8000/analyze/Technology

# Check service health
curl http://localhost:8000/health

# Get available sectors
curl http://localhost:8000/sectors
```

## Deployment on Render

### 1. Prepare for Deployment

Ensure your project has these files:
- `cd.py` (your main application)
- `index.html` (your frontend)
- `requirements.txt` (dependencies)

### 2. Create Render Web Service

1. **Sign up** at [render.com](https://render.com)
2. **Connect your GitHub repository**
3. **Create a new Web Service**

### 3. Configure Deployment

**Build Settings**:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python cd.py`

**Environment Variables**:
```
GROQ_API_KEY=your_groq_api_key
SERPAPI_KEY=your_serpapi_key
```

### 4. Deploy

Render will automatically deploy your application and provide a URL like:
`https://your-app-name.onrender.com`

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for AI analysis | Yes |
| `SERPAPI_KEY` | SerpAPI key for web search | Yes |
| `PORT` | Server port (auto-set by Render) | No |

## Usage

### Web Interface

1. **Open the application** in your browser
2. **Enter a sector name** (e.g., "Technology", "Healthcare", "Banking")
3. **Click "Generate Investment Report"**
4. **Review the comprehensive analysis** with:
   - Executive Summary
   - Market Analysis
   - Investment Opportunities
   - Risk Assessment
   - Specific Recommendations

### Supported Sectors

The platform works with any sector, but is optimized for:
- Technology
- Healthcare
- Banking
- Automotive
- Renewable Energy
- FMCG
- Infrastructure
- Pharmaceuticals
- Real Estate
- Telecommunications

## Features Detail

### Real-time Data Collection
- Searches multiple sources for current market information
- Scrapes financial news, company reports, and market analysis
- Combines data from 6+ sources per analysis

### AI-Powered Analysis
- Uses advanced language models for intelligent analysis
- Generates specific investment recommendations
- Provides risk assessment and mitigation strategies
- Creates actionable insights for investors

### Professional Reporting
- Institutional-quality investment reports
- Specific buy/sell/hold recommendations
- Target allocation percentages
- Timeline-based return expectations

## Rate Limits

- **5 requests per minute** per IP address
- **10 requests per hour** per IP address
- Designed to prevent abuse while allowing legitimate use

## Troubleshooting

### Common Issues

1. **"Missing API Keys" Error**:
   - Ensure `.env` file exists with valid API keys
   - Check that environment variables are set correctly

2. **"Search service unavailable"**:
   - Verify SerpAPI key is valid and has remaining quota
   - Check internet connectivity

3. **"Analysis failed"**:
   - Verify Groq API key is valid
   - Check if you've exceeded API rate limits

4. **Frontend not loading**:
   - Ensure `index.html` is in the same directory as `cd.py`
   - Check browser console for JavaScript errors

### Logs

The application provides detailed logging. Check the console output for:
- Service startup information
- API health status
- Analysis progress
- Error messages

## Development

### Adding New Features

1. **Backend changes**: Modify `cd.py`
2. **Frontend changes**: Modify `index.html`
3. **New dependencies**: Add to `requirements.txt`

### Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test analysis endpoint
curl http://localhost:8000/analyze/Technology

# Check logs for errors
python cd.py
```

## Security

- Input validation and sanitization
- Rate limiting to prevent abuse
- CORS middleware for secure cross-origin requests
- No sensitive data stored in frontend

## Performance

- Async processing for concurrent operations
- Connection pooling for efficient web scraping
- Content compression (GZip)
- Optimized scraping with timeouts and limits

## License

This project is for educational and research purposes. Ensure you comply with API terms of service and rate limits.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check service health at `/health`

## Contributing

To contribute:
1. Fork the project
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Note**: This application generates investment analysis for informational purposes only. It does not constitute financial advice. Always consult with qualified financial advisors before making investment decisions.
