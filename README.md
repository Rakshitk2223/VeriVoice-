# VeriVoice News: AI-Powered Fact-Checker for the Visually Impaired

A web-based voice assistant that allows visually impaired users to listen to news summaries and receive voice-based credibility assessments using AI.

## Project Structure

```
VeriVoice/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env               # API configuration (NewsAPI key)
├── .gitignore         # Git ignore file
└── templates/
    └── index.html     # Frontend interface
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key:**
   - Get a free API key from [NewsAPI.org](https://newsapi.org/register)
   - Update `.env` file with your API key

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   - Go to http://localhost:5000
   - Use voice commands or type news queries

## Features

- 🎤 **Voice Recognition**: Speak commands like "news about technology"
- ⌨️ **Text Input**: Type commands as backup option
- 📰 **News Summarization**: Get concise 3-sentence summaries
- 🔍 **Credibility Check**: AI-powered fake news detection (Phase 2)
- ♿ **Accessibility**: Designed for visually impaired users

## Usage Examples

- "Read news about sports"
- "Latest technology news"
- "Climate change news"

The app will fetch recent articles, generate summaries, and read them aloud with credibility assessments.
