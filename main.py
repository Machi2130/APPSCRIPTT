from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import asyncio
import httpx
import aiocache
from aiocache import Cache
from aiocache.serializers import JsonSerializer
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from pydantic import BaseModel, field_validator
import os
import sys
from contextlib import asynccontextmanager
from groq import Groq
from serpapi import GoogleSearch
import re
from bs4 import BeautifulSoup
import requests
import trafilatura
from urllib.parse import urlparse
import hashlib
import json
from functools import lru_cache, wraps
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Python 3.13 compatibility check
print(f"Python version: {sys.version}")
print(f"Python version info: {sys.version_info}")

load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

if not GROQ_API_KEY or not SERPAPI_KEY:
    print("Missing required API keys in environment variables")
    print("Please set GROQ_API_KEY and SERPAPI_KEY in your .env file or environment")
    # Don't exit on production, let it continue for health checks
    if os.getenv("RENDER"):
        print("Running on Render - API keys must be set in environment variables")
    else:
        exit(1)

# Constants
GROQ_MODEL = "deepseek-r1-distill-llama-70b"
TOP_URLS_TO_PICK = 6
CONTENT_LIMIT = 3000
MAX_CONCURRENT_SCRAPES = 5
CACHE_TTL = 3600
REQUEST_TIMEOUT = 15
MAX_WORKERS = 4

# Setup logging with better formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize clients with better error handling
try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    groq_client = None

thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Try to import crawl4ai, but make it optional
try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
    logger.info("crawl4ai is available")
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.info("crawl4ai not available, using BeautifulSoup and trafilatura")

# Enhanced input sanitization
def sanitize_input(text: str, max_length: int = 1000, preserve_markdown: bool = False) -> str:
    if not text:
        return ""
    
    text = text[:max_length]
    
    if not preserve_markdown:
        text = re.sub(r'[<>"\'`#*|]', '', text)
    else:
        text = re.sub(r'[<>"\'`]', '', text)
    
    text = re.sub(r'\s+', ' ', text) if not preserve_markdown else re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# Input validation
class AnalysisRequest(BaseModel):
    sector: str
    
    @field_validator('sector')
    @classmethod
    def validate_sector(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Sector name must be at least 2 characters long")
        if len(v.strip()) > 50:
            raise ValueError("Sector name must be less than 50 characters")
        if not re.match(r'^[a-zA-Z0-9\s\-_&]+$', v):
            raise ValueError("Sector name contains invalid characters")
        
        dangerous_patterns = ['<script', 'javascript:', 'data:', '../', 'DROP', 'SELECT']
        v_lower = v.lower()
        if any(pattern in v_lower for pattern in dangerous_patterns):
            raise ValueError("Invalid sector name")
        
        return v.strip()

class ScrapedContent(BaseModel):
    url: str
    title: str
    content: str
    source: str
    scraped_at: str
    content_length: int

# Enhanced search functionality
class OptimizedSearcher:
    @staticmethod
    async def search_and_pick_urls(sector: str) -> List[str]:
        logger.info(f"Searching for '{sector}' sector information")
        
        def _search():
            clean_sector = sanitize_input(sector, 100)
            
            params = {
                "engine": "duckduckgo",
                "q": f"{clean_sector} India market trend and investment chances",
                "kl": "in-en",
                "api_key": SERPAPI_KEY,
                "num": TOP_URLS_TO_PICK + 3
            }
            
            search = GoogleSearch(params)
            return search.get_dict()
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(thread_pool, _search)
            
            urls = set()
            
            # Process organic results
            for item in results.get("organic_results", [])[:TOP_URLS_TO_PICK]:
                link = item.get("link")
                if link and not link.lower().endswith(('.pdf', '.doc', '.docx')):
                    urls.add(link)
            
            # Process news results
            for item in results.get("news_results", [])[:3]:
                link = item.get("link")
                if link and not link.lower().endswith(('.pdf', '.doc', '.docx')):
                    urls.add(link)
            
            unique_urls = list(urls)[:TOP_URLS_TO_PICK]
            
            if unique_urls:
                logger.info(f"Found {len(unique_urls)} URLs")
                return unique_urls
            else:
                raise Exception("No URLs found")
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Search service unavailable: {str(e)}")

# Enhanced web scraping using httpx instead of aiohttp
class OptimizedWebScraper:
    @staticmethod
    async def scrape_websites(urls: List[str]) -> List[ScrapedContent]:
        logger.info(f"Scraping {len(urls)} websites")
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
        
        async def scrape_single_url(client: httpx.AsyncClient, url: str) -> Optional[ScrapedContent]:
            async with semaphore:
                return await OptimizedWebScraper._scrape_url_optimized(client, url)
        
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(REQUEST_TIMEOUT),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                headers={'User-Agent': 'Mozilla/5.0 (compatible; Research Bot)'}
            ) as client:
                tasks = [scrape_single_url(client, url) for url in urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            scraped_contents = []
            for result in results:
                if isinstance(result, ScrapedContent):
                    scraped_contents.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Scraping error: {str(result)}")
            
            logger.info(f"Successfully scraped {len(scraped_contents)} out of {len(urls)} websites")
            
            if not scraped_contents:
                raise HTTPException(status_code=503, detail="No content could be scraped")
            
            return scraped_contents
            
        except Exception as e:
            logger.error(f"Web scraping failed: {str(e)}")
            raise HTTPException(status_code=503, detail="No content could be scraped")
    
    @staticmethod
    async def _scrape_url_optimized(client: httpx.AsyncClient, url: str) -> Optional[ScrapedContent]:
        if url.lower().endswith(('.pdf', '.doc', '.docx', '.ppt')):
            return None
        
        # Try httpx-based scraping first
        try:
            result = await OptimizedWebScraper._scrape_with_httpx(client, url)
            if result and len(result.get('content', '')) > 100:
                scraped_content = OptimizedWebScraper._create_scraped_content(url, result)
                logger.info(f"Scraped {urlparse(url).netloc} using httpx")
                return scraped_content
        except Exception as e:
            logger.error(f"HTTPX scraping failed for {urlparse(url).netloc}: {str(e)}")
        
        # Fallback methods
        fallback_methods = [
            OptimizedWebScraper._scrape_with_trafilatura,
        ]
        
        if CRAWL4AI_AVAILABLE:
            fallback_methods.insert(0, OptimizedWebScraper._scrape_with_crawl4ai)
        
        for method in fallback_methods:
            try:
                result = await method(url)
                if result and len(result.get('content', '')) > 100:
                    scraped_content = OptimizedWebScraper._create_scraped_content(url, result)
                    logger.info(f"Scraped {urlparse(url).netloc} using {result['method']}")
                    return scraped_content
            except Exception:
                continue
        
        return None
    
    @staticmethod
    async def _scrape_with_httpx(client: httpx.AsyncClient, url: str) -> Optional[Dict]:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                html = response.text
                
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    thread_pool,
                    OptimizedWebScraper._parse_html_optimized,
                    html
                )
                
                result['method'] = 'httpx'
                return result
        except Exception as e:
            logger.error(f"HTTPX scraping failed: {str(e)}")
        
        return None
    
    @staticmethod
    def _parse_html_optimized(html: str) -> Dict:
        soup = BeautifulSoup(html, 'lxml')
        
        for tag in ['script', 'style', 'nav', 'footer', 'header', 'aside']:
            for element in soup.find_all(tag):
                element.decompose()
        
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else "No Title"
        title = sanitize_input(title, 200)
        
        content_selectors = [
            'article', 'main', '[role="main"]', 
            '.content', '.post-content', '.entry-content'
        ]
        
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join(el.get_text(separator=' ', strip=True) for el in elements)
                break
        
        if not content:
            paragraphs = soup.find_all('p', limit=50)
            content = ' '.join(p.get_text(strip=True) for p in paragraphs)
        
        content = sanitize_input(content, CONTENT_LIMIT)
        
        return {
            'title': title,
            'content': content[:CONTENT_LIMIT]
        }
    
    @staticmethod
    async def _scrape_with_crawl4ai(url: str) -> Optional[Dict]:
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url, word_count_threshold=50, bypass_cache=True)
                if result.success and result.markdown:
                    title_match = re.search(r'^#\s+(.+)', result.markdown, re.MULTILINE)
                    title = title_match.group(1) if title_match else "No Title"
                    title = sanitize_input(title, 200)
                    
                    content = sanitize_input(result.markdown, CONTENT_LIMIT)
                    
                    return {
                        'title': title,
                        'content': content[:CONTENT_LIMIT],
                        'method': 'crawl4ai'
                    }
        except Exception as e:
            logger.error(f"Crawl4AI failed: {str(e)}")
        
        return None
    
    @staticmethod
    async def _scrape_with_trafilatura(url: str) -> Optional[Dict]:
        def _extract():
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; Research Bot)'}
            response = requests.get(url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                content = trafilatura.extract(response.text, include_comments=False)
                if content:
                    soup = BeautifulSoup(response.text, 'lxml')
                    title_tag = soup.find('title')
                    title = title_tag.get_text().strip() if title_tag else "No Title"
                    
                    return {
                        'title': sanitize_input(title, 200),
                        'content': sanitize_input(content, CONTENT_LIMIT),
                        'method': 'trafilatura'
                    }
            return None
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(thread_pool, _extract)
        except Exception as e:
            logger.error(f"Trafilatura failed: {str(e)}")
        
        return None
    
    @staticmethod
    def _create_scraped_content(url: str, result: Dict) -> ScrapedContent:
        source = urlparse(url).netloc
        return ScrapedContent(
            url=url,
            title=result['title'],
            content=result['content'],
            source=source,
            scraped_at=datetime.now().isoformat(),
            content_length=len(result['content'])
        )

# Enhanced LLM processing for detailed investment analysis
class OptimizedLLMProcessor:
    @staticmethod
    async def generate_detailed_investment_report(scraped_data: List[ScrapedContent], sector: str) -> str:
        logger.info(f"Generating detailed investment report for {sector}")
        
        # Check if groq_client is available
        if not groq_client:
            logger.error("Groq client not available")
            return OptimizedLLMProcessor._generate_fallback_report(sector, len(scraped_data))
        
        def _generate_report():
            clean_sector = sanitize_input(sector, 100)
            
            # Prepare content for LLM
            content_chunks = []
            total_length = 0
            
            for i, content in enumerate(scraped_data, 1):
                if total_length > 15000:  # Prevent token limit issues
                    break
                
                chunk = (f"SOURCE {i} - {content.source}:\n"
                        f"Title: {content.title[:200]}\n"
                        f"Content: {content.content[:1200]}\n"
                        f"{'='*50}\n\n")
                
                content_chunks.append(chunk)
                total_length += len(chunk)
            
            market_data = f"COMPREHENSIVE MARKET DATA FOR {clean_sector.upper()} SECTOR (INDIA FOCUS):\n\n" + ''.join(content_chunks)
            
            # Enhanced prompt for detailed investment analysis
            report_timestamp = datetime.now()
            report_id = f"INV-{clean_sector.upper()}-{report_timestamp.strftime('%Y%m%d-%H%M')}"
            
            prompt = f"""You are a senior investment analyst creating a comprehensive investment report for institutional investors and fund managers. Generate a detailed, actionable investment analysis for the {clean_sector} sector in India.

MARKET DATA:
{market_data}

Create a professional investment report with the following structure:

# {clean_sector.title()} Sector Investment Analysis Report

**Report ID:** {report_id}  
**Date:** {report_timestamp.strftime("%B %d, %Y")}  
**Analyst:** AI Investment Research Team  

## Executive Summary
- **Investment Thesis:** [2-3 key investment arguments]
- **Sector Outlook:** [Bullish/Neutral/Bearish with reasoning]
- **Target Allocation:** [Recommended portfolio allocation %]
- **Investment Horizon:** [Short/Medium/Long term recommendation]

## Market Overview & Size
- **Current Market Size:** [In Crores/USD if available]
- **Growth Rate (CAGR):** [Historical and projected]
- **Market Position:** [India's global ranking in this sector]
- **Key Market Drivers:** [Top 3-4 growth catalysts]

## Industry Structure & Key Players
- **Market Leaders:** [Top 5 companies with market share]
- **Competitive Landscape:** [Fragmented/Consolidated analysis]
- **Emerging Players:** [New entrants and disruptors]
- **Value Chain Analysis:** [Key segments and profit pools]

## Financial Performance Analysis
- **Revenue Growth:** [Sector-wide growth trends]
- **Profitability Metrics:** [EBITDA margins, ROE, ROIC trends]
- **Debt Levels:** [Leverage analysis]
- **Cash Flow Generation:** [Free cash flow trends]
- **Valuation Metrics:** [P/E, EV/EBITDA vs historical averages]

## Investment Opportunities
### Primary Investment Themes:
1. **[Theme 1]:** [Detailed explanation with expected returns]
2. **[Theme 2]:** [Market size and growth potential]
3. **[Theme 3]:** [Regulatory tailwinds or structural changes]

### Specific Investment Ideas:
- **Large Cap Plays:** [2-3 established companies with rationale]
- **Mid Cap Opportunities:** [2-3 growth stories]
- **Thematic ETFs/Mutual Funds:** [Sector-specific investment vehicles]

## Risk Analysis
### Key Risks:
1. **Regulatory Risk:** [Policy changes, compliance costs]
2. **Economic Risk:** [GDP growth, interest rates, inflation impact]
3. **Competitive Risk:** [New entrants, technology disruption]
4. **Operational Risk:** [Supply chain, raw material costs]
5. **ESG Risk:** [Environmental, social, governance concerns]

### Risk Mitigation Strategies:
- [Diversification approaches]
- [Hedging mechanisms]
- [Exit triggers and stop-loss levels]

## Sector Catalysts & Tailwinds
- **Government Initiatives:** [PLI schemes, policy support]
- **Technology Adoption:** [Digital transformation, automation]
- **Demographic Trends:** [Consumer behavior changes]
- **Infrastructure Development:** [Supporting ecosystem growth]
- **Export Potential:** [Global market opportunities]

## Investment Recommendation

### Overall Rating: [BUY/HOLD/SELL]

**Rationale:**
[2-3 paragraphs explaining the investment decision with specific reasoning]

**Target Returns:**
- **1 Year:** [Expected returns with probability]
- **3 Year:** [Medium-term outlook]
- **5 Year:** [Long-term potential]

**Allocation Strategy:**
- **Conservative Portfolio:** [X% allocation]
- **Moderate Portfolio:** [Y% allocation]
- **Aggressive Portfolio:** [Z% allocation]

## Action Items for Investors
1. **Immediate Actions:** [What to do in next 30 days]
2. **Monitoring Metrics:** [Key indicators to track]
3. **Review Schedule:** [When to reassess the investment thesis]
4. **Exit Strategy:** [Clear exit criteria and triggers]

## Key Data Sources
- Market research reports and industry analysis
- Company financial statements and investor presentations
- Government policy documents and regulatory filings
- News and market commentary from credible sources

---
**Disclaimer:** This report is for informational purposes only and does not constitute investment advice. Past performance does not guarantee future results. Investors should conduct their own research and consult with financial advisors before making investment decisions.

Generate this report using the provided market data, ensuring all analysis is specific, actionable, and based on the scraped information. Use actual numbers, company names, and specific details from the sources wherever possible."""

            # Generate with Groq
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a senior investment analyst with 15+ years of experience in Indian equity markets. Create detailed, actionable investment reports that institutional investors can use for decision-making. Always provide specific recommendations with clear rationale."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent analysis
                max_tokens=4000   # Increased for detailed reports
            )
            
            return completion.choices[0].message.content
        
        try:
            loop = asyncio.get_event_loop()
            report = await loop.run_in_executor(thread_pool, _generate_report)
            
            if report:
                # Remove dangerous patterns while preserving markdown
                dangerous_patterns = ['<script', 'javascript:', 'data:text/html']
                for pattern in dangerous_patterns:
                    report = report.replace(pattern, '[REMOVED]')
            
            logger.info(f"Generated detailed investment report for {sector}")
            return report or OptimizedLLMProcessor._generate_fallback_report(sector, len(scraped_data))
            
        except Exception as e:
            logger.error(f"LLM processing failed: {str(e)}")
            return OptimizedLLMProcessor._generate_fallback_report(sector, len(scraped_data))
    
    @staticmethod
    def _generate_fallback_report(sector: str, source_count: int) -> str:
        timestamp = datetime.now()
        return f"""# {sector.title()} Sector Investment Analysis Report

**Report ID:** FALLBACK-{sector.upper()}-{timestamp.strftime('%Y%m%d-%H%M')}  
**Date:** {timestamp.strftime("%B %d, %Y")}  

## Executive Summary
- Investment analysis attempted for {sector} sector using {source_count} data sources
- Service temporarily limited - detailed analysis requires retry
- Sector demonstrates significant importance in India's economic landscape

## Current Status
- **Data Sources Accessed:** {source_count} industry websites
- **Analysis Status:** Partial due to service limitations
- **Recommendation:** Retry analysis or consult with financial advisors for complete assessment

## Next Steps
1. **Retry Analysis:** Refresh the report for complete insights
2. **Manual Research:** Supplement with direct company and industry research  
3. **Professional Consultation:** Engage with certified financial advisors

---
**Generated:** {timestamp.strftime("%B %d, %Y at %I:%M %p")}  
**Status:** Fallback Report - Requires Retry for Complete Analysis
"""

# Main analysis flow
class OptimizedAnalysisFlow:
    @staticmethod
    async def run_full_analysis(sector: str) -> Dict:
        start_time = time.time()
        logger.info(f"Starting comprehensive analysis for {sector}")
        
        try:
            # Step 1: Search
            search_start = time.time()
            top_urls = await OptimizedSearcher.search_and_pick_urls(sector)
            search_time = time.time() - search_start
            
            # Step 2: Scrape
            scrape_start = time.time()
            scraped_data = await OptimizedWebScraper.scrape_websites(top_urls)
            scrape_time = time.time() - scrape_start
            
            # Step 3: Generate detailed report
            llm_start = time.time()
            report = await OptimizedLLMProcessor.generate_detailed_investment_report(scraped_data, sector)
            llm_time = time.time() - llm_start
            
            total_time = time.time() - start_time
            
            result = {
                "sector": sector,
                "report": report,
                "metadata": {
                    "urls_found": len(top_urls),
                    "sites_scraped": len(scraped_data),
                    "total_processing_time": round(total_time, 2),
                    "search_time": round(search_time, 2),
                    "scrape_time": round(scrape_time, 2),
                    "analysis_time": round(llm_time, 2),
                    "timestamp": datetime.now().isoformat(),
                    "report_length": len(report),
                    "sources": [
                        {
                            "url": content.url,
                            "title": content.title[:100],
                            "source": content.source,
                            "content_length": content.content_length
                        }
                        for content in scraped_data
                    ]
                },
                "cached": False,
                "response_time": round(total_time, 2)
            }
            
            logger.info(f"Analysis completed for {sector} in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise e

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Investment Analysis API")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"GROQ API configured: {'Yes' if GROQ_API_KEY else 'No'}")
    logger.info(f"SerpAPI configured: {'Yes' if SERPAPI_KEY else 'No'}")
    logger.info(f"Crawl4AI available: {'Yes' if CRAWL4AI_AVAILABLE else 'No'}")
    logger.info(f"Running on Render: {'Yes' if os.getenv('RENDER') else 'No'}")
    logger.info("Using HTTPX for HTTP requests (Python 3.13.3 compatible)")
    
    yield
    
    logger.info("Shutting down API")
    thread_pool.shutdown(wait=True)

app = FastAPI(
    title="Investment Analysis API",
    description="Professional investment research with detailed sector analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600
)

# Rate limiting (simple in-memory implementation)
request_counts = {}
def check_rate_limit(client_ip: str, limit: int = 10, window: int = 60) -> bool:
    current_time = time.time()
    
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    # Clean old requests
    request_counts[client_ip] = [
        t for t in request_counts[client_ip] 
        if current_time - t < window
    ]
    
    if len(request_counts[client_ip]) >= limit:
        return False
    
    request_counts[client_ip].append(current_time)
    return True

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as cd.py</p>",
            status_code=404
        )

@app.get("/analyze/{sector}")
async def analyze_sector(sector: str, request: Request):
    """Generate comprehensive investment analysis for a sector"""
    
    # Check if API keys are available
    if not GROQ_API_KEY or not SERPAPI_KEY:
        raise HTTPException(
            status_code=503, 
            detail="API keys not configured. Please set GROQ_API_KEY and SERPAPI_KEY environment variables."
        )
    
    # Rate limiting
    client_ip = request.client.host
    if not check_rate_limit(client_ip, limit=5, window=60):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    
    # Input validation
    try:
        validated_request = AnalysisRequest(sector=sector)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        # Run analysis
        result = await OptimizedAnalysisFlow.run_full_analysis(validated_request.sector)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "groq_api": bool(groq_client and GROQ_API_KEY),
            "serpapi": bool(SERPAPI_KEY),
            "crawl4ai": CRAWL4AI_AVAILABLE
        },
        "features": {
            "detailed_investment_analysis": True,
            "real_time_data_scraping": True,
            "professional_reporting": True,
            "httpx_client": True
        },
        "environment": {
            "render": bool(os.getenv("RENDER")),
            "python_version": sys.version_info[:3]
        }
    }

@app.get("/sectors")
def get_available_sectors():
    """Get list of recommended sectors for analysis"""
    return {
        "sectors": [
            {"name": "Technology", "description": "IT services, software, and technology companies"},
            {"name": "Healthcare", "description": "Pharmaceuticals, hospitals, and medical devices"},
            {"name": "Banking", "description": "Commercial banks, NBFCs, and financial services"},
            {"name": "Automotive", "description": "Car manufacturers, auto components, and EV"},
            {"name": "Renewable Energy", "description": "Solar, wind, and clean energy companies"},
            {"name": "FMCG", "description": "Fast-moving consumer goods and brands"},
            {"name": "Infrastructure", "description": "Construction, roads, and urban development"},
            {"name": "Pharmaceuticals", "description": "Drug manufacturing and biotechnology"},
            {"name": "Real Estate", "description": "Property development and real estate investment"},
            {"name": "Telecommunications", "description": "Telecom services and digital infrastructure"},
            {"name": "Steel", "description": "Steel production and metal industries"},
            {"name": "Energy", "description": "Oil, gas, and traditional energy sectors"},
            {"name": "Agriculture", "description": "Farming, food processing, and agri-tech"},
            {"name": "Textiles", "description": "Clothing, fabric, and textile manufacturing"},
            {"name": "Aviation", "description": "Airlines, airports, and aviation services"},
            {"name": "E-commerce", "description": "Online retail and digital marketplaces"},
            {"name": "Fintech", "description": "Digital payments and financial technology"},
            {"name": "Cement", "description": "Cement production and building materials"}
        ],
        "total": 18,
        "last_updated": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested resource was not found",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Render sets PORT environment variable
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"  # Important: must bind to 0.0.0.0 for Render
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Python version: {sys.version}")
    logger.info("Using HTTPX instead of aiohttp for Python 3.13.3 compatibility")
    logger.info(f"Access the API docs at http://localhost:{port}/docs" if port == 8000 else f"API will be available at your deployment URL")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        access_log=False,
        server_header=False
    )
