from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import requests
import re

# Try to import optional libraries with fallback
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    print("⚠️  joblib not installed. ML features will be limited.")
    JOBLIB_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️  NLTK not installed. Using basic text processing.")
    NLTK_AVAILABLE = False

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    SUMY_AVAILABLE = True
    print("✓ Sumy library available")
except ImportError:
    print("⚠️  Sumy not installed. Using basic summarization.")
    SUMY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-learn not installed. ML features will be limited.")
    SKLEARN_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Debug: Print API key status (without revealing the actual key)
if NEWS_API_KEY and NEWS_API_KEY != 'YOUR_NEWS_API_KEY_HERE':
    print("✓ NewsAPI key is configured")
else:
    print("⚠️  WARNING: NewsAPI key is not configured! Please update your .env file with a valid API key.")
    print("   Get your free API key from: https://newsapi.org/register")

# Download NLTK data if NLTK is available
if NLTK_AVAILABLE:
    try:
        import nltk
        # Check and download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ NLTK punkt tokenizer found")
        except LookupError:
            print("📦 Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
            print("✓ NLTK stopwords found")
        except LookupError:
            print("📦 Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            
        # Additional NLTK data that Sumy might need
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("📦 Downloading additional NLTK data...")
            nltk.download('punkt_tab', quiet=True)
            
    except Exception as e:
        print(f"⚠️  NLTK setup warning: {e}")
        NLTK_AVAILABLE = False

# Test Sumy import with better error handling
if SUMY_AVAILABLE:
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer
        from sumy.nlp.stemmers import Stemmer
        
        # Skip the test at startup - it might cause issues
        # We'll test Sumy when actually needed during summarization
        print("✓ Sumy library imported successfully")
        
    except ImportError as e:
        print(f"❌ Error importing Sumy: {e}")
        SUMY_AVAILABLE = False
    except Exception as e:
        print(f"⚠️  Sumy setup warning: {e}")
        print("   Sumy imported but may have issues. Will use fallback summarization.")
        SUMY_AVAILABLE = False

# Global ML Model Loading (Placeholders for Phase 2)
tfidf_vectorizer = None
fake_news_model = None

def load_ml_models():
    """
    Load ML models from .pkl files for fake news detection.
    This will be implemented in Phase 2. For now, these are placeholders.
    """
    global tfidf_vectorizer, fake_news_model
    if not JOBLIB_AVAILABLE:
        print("⚠️  joblib not available. ML models disabled.")
        tfidf_vectorizer = None
        fake_news_model = None
        return
        
    try:
        import joblib
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        fake_news_model = joblib.load('logistic_regression_model.pkl')
        print("ML models loaded successfully")
    except FileNotFoundError:
        print("ML model files not found. Using placeholder models for Phase 1.")
        # Placeholder objects for Phase 1
        tfidf_vectorizer = None
        fake_news_model = None
    except Exception as e:
        print(f"Error loading ML models: {e}")
        tfidf_vectorizer = None
        fake_news_model = None

# Load models at startup
load_ml_models()

def predict_fake_news(text):
    """
    Predict if news is fake or real using ML models.
    This is a placeholder for Phase 2 - currently returns unknown credibility.
    """
    # Phase 2: This will contain the ML model logic
    # For now, return placeholder verdict
    return "unknown credibility"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/test')
def test():
    """Test endpoint to verify the API is working"""
    return jsonify({
        'status': 'success',
        'message': 'Flask app is running!',
        'api_key_configured': bool(NEWS_API_KEY and NEWS_API_KEY != 'YOUR_NEWS_API_KEY_HERE')
    })

@app.route('/get_news_summary', methods=['POST'])
def get_news_summary():
    """
    Get news summary based on voice command and provide fake news detection
    """
    try:
        # Check if API key is configured
        if not NEWS_API_KEY or NEWS_API_KEY == 'YOUR_NEWS_API_KEY_HERE':
            print("❌ ERROR: NewsAPI key not configured")
            return jsonify({
                'summary': 'Error: NewsAPI key not configured. Please add your API key to the .env file.',
                'status': 'error',
                'fake_news_verdict': 'N/A'
            })
        
        # Get JSON data from request
        data = request.json
        command = data.get('command', '').lower()
        print(f"📝 Received command: '{command}'")
        
        # Extract search query from command
        query = "world news"  # default
        if "about" in command:
            # Extract text after "about"
            about_index = command.find("about")
            if about_index != -1:
                query = command[about_index + 5:].strip()
        elif "news" in command:
            # Look for keywords after "news"
            news_index = command.find("news")
            if news_index != -1:
                remaining = command[news_index + 4:].strip()
                if remaining and not remaining.startswith("about"):
                    query = remaining
        
        # If query is still generic, try to extract any meaningful content
        if query == "world news" and len(command.split()) > 2:
            # Remove common words and use remaining as query
            common_words = ['read', 'news', 'about', 'tell', 'me', 'get', 'latest']
            words = [word for word in command.split() if word not in common_words]
            if words:
                query = ' '.join(words)
        
        print(f"🔍 Search query: '{query}'")
        
        # Construct NewsAPI URL
        news_url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': NEWS_API_KEY,
            'pageSize': 5,
            'language': 'en',
            'sortBy': 'publishedAt'
        }
        
        print(f"🌐 Making request to NewsAPI...")
        
        # Make request to NewsAPI
        response = requests.get(news_url, params=params, timeout=10)
        print(f"📡 API Response Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ ERROR: Invalid API key")
            return jsonify({
                'summary': 'Error: Invalid NewsAPI key. Please check your API key in the .env file.',
                'status': 'error',
                'fake_news_verdict': 'N/A'
            })
        elif response.status_code == 429:
            print("❌ ERROR: API rate limit exceeded")
            return jsonify({
                'summary': 'Error: API rate limit exceeded. Please try again later.',
                'status': 'error',
                'fake_news_verdict': 'N/A'
            })
        
        response.raise_for_status()
        
        response_data = response.json()
        articles = response_data.get('articles', [])
        total_results = response_data.get('totalResults', 0)
        
        print(f"📰 Found {total_results} articles, received {len(articles)} articles")
        
        if total_results == 0:
            print(f"⚠️  No articles found for query: '{query}'")
            return jsonify({
                'summary': f'Sorry, I could not find any news articles about "{query}". Try a different topic.',
                'status': 'error',
                'fake_news_verdict': 'N/A'
            })
        
        # Find first article with meaningful content
        selected_article = None
        article_title = None
        for i, article in enumerate(articles):
            content = article.get('content') or article.get('description', '')
            title = article.get('title', '')
            print(f"📄 Article {i+1}: {title[:50]}... Content length: {len(content) if content else 0}")
            
            if content and len(content.strip()) > 50:  # Ensure meaningful content
                selected_article = content
                article_title = title
                print(f"✓ Selected article: {title[:50]}...")
                break
        
        if not selected_article:
            print("❌ No suitable articles found with sufficient content")
            return jsonify({
                'summary': f'Sorry, I could not find detailed news articles about "{query}". The available articles don\'t have enough content to summarize.',
                'status': 'error',
                'fake_news_verdict': 'N/A'
            })
        
        print(f"📝 Processing article: {article_title[:50]}...")
        
        # Clean the article text
        cleaned_text = re.sub(r'[^\w\s]', '', selected_article.lower())
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        print(f"🔧 Generating summary...")
        
        # Try to use Sumy for summarization, with fallback
        summary = None
        if SUMY_AVAILABLE:
            try:
                # Generate summary using Sumy
                parser = PlaintextParser.from_string(selected_article, Tokenizer('english'))
                stemmer = Stemmer('english')
                summarizer = LsaSummarizer(stemmer)
                summarizer.stop_words = ['english']
                
                # Generate 3-sentence summary
                summary_sentences = summarizer(parser.document, 3)
                summary = ' '.join(str(sentence) for sentence in summary_sentences)
                print(f"✓ Sumy summary generated: {len(summary)} characters")
                
            except Exception as sumy_error:
                print(f"⚠️  Sumy error during summarization: {sumy_error}")
                print("🔄 Using fallback summarization...")
                summary = None  # Force fallback
        
        # Fallback summarization if Sumy failed or unavailable
        if not summary or not summary.strip():
            print("🔄 Using simple extractive summarization...")
            
            # Simple extractive summary
            sentences = selected_article.split('. ')
            # Take first 3 sentences or up to 300 characters
            if len(sentences) >= 3:
                summary = '. '.join(sentences[:3]) + '.'
            else:
                summary = selected_article[:300] + "..." if len(selected_article) > 300 else selected_article
            print(f"✓ Simple summary generated: {len(summary)} characters")
        
        if not summary or not summary.strip():
            # Last resort fallback
            summary = selected_article[:300] + "..." if len(selected_article) > 300 else selected_article
            print(f"⚠️  Using truncated article as summary: {len(summary)} characters")
        
        print(f"✓ Summary generated: {len(summary)} characters")
        
        # Get fake news prediction (placeholder for Phase 2)
        fake_news_verdict = predict_fake_news(cleaned_text)
        
        print(f"✅ Request completed successfully")
        
        return jsonify({
            'summary': summary,
            'status': 'success',
            'fake_news_verdict': fake_news_verdict
        })
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {str(e)}")
        return jsonify({
            'summary': 'Sorry, there was a network error fetching the news. Please check your internet connection and try again.',
            'status': 'error',
            'fake_news_verdict': 'N/A'
        })
    except Exception as e:
        print(f"❌ Unexpected error in get_news_summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'summary': 'Sorry, there was an unexpected error processing your request.',
            'status': 'error',
            'fake_news_verdict': 'N/A'
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
