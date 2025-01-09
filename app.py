import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from newsapi.newsapi_client import NewsApiClient
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from collections import Counter
import base64
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os
import ssl

# Handle SSL certificate verification issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Create NLTK data directory if it doesn't exist
nltk_data_dir = os.path.expanduser(r'\Advanced-News-Sentiment-Analyzer\nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download required NLTK data with explicit downloading
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        # Download punkt_tab 
        nltk.download('punkt_tab', quiet=True) # Add this line 
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        return False

# Initialize NLTK components
@st.cache_resource
def initialize_nltk():
    """Initialize NLTK components"""
    if download_nltk_data():
        return SentimentIntensityAnalyzer(), set(stopwords.words('english'))
    return None, None

# Configure page settings
st.set_page_config(
    page_title="Advanced News Sentiment Analyzer",
    page_icon="📰",
    layout="wide"
)

# Custom CSS with dark mode support
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #F44336;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #9E9E9E;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .headline-card {
        background-color: #2D2D2D;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .source-tag {
        background-color: #424242;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    .date-tag {
        color: #9E9E9E;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame()
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.now()

def extract_keywords(text, n=1):
    """Extract keywords from text using NLTK"""
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Tokenize and remove stopwords
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_freq = Counter(words)
    return dict(word_freq.most_common(n))

def create_wordcloud(text):
    """Create and return wordcloud image"""
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='black',
        colormap='viridis'
    ).generate(text)
    
    return wordcloud

def get_sentiment_color(score):
    """Return color based on sentiment score"""
    if score > 0.05:
        return "sentiment-positive"
    elif score < -0.05:
        return "sentiment-negative"
    else:
        return "sentiment-neutral"

def get_sentiment_label(score):
    """Convert sentiment score to label with emoji"""
    if score > 0.05:
        return "Positive 📈"
    elif score < -0.05:
        return "Negative 📉"
    else:
        return "Neutral ➖"

def fetch_news(api_key, category, country_code, num_headlines, search_method, date_range=None):
    """Enhanced news fetching with date range support"""
    try:
        newsapi = NewsApiClient(api_key=api_key)
        
        if search_method == "Top Headlines":
            response = newsapi.get_top_headlines(
                category=category if category != 'general' else None,
                country=country_code,
                page_size=num_headlines
            )
        else:
            # Calculate date range
            if date_range:
                from_date = (datetime.now() - timedelta(days=date_range)).strftime('%Y-%m-%d')
            else:
                from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                
            response = newsapi.get_everything(
                q=f"country {selected_country}" if category == 'general' else category,
                from_param=from_date,
                language='en',
                sort_by='publishedAt',
                page_size=num_headlines
            )
        
        return response.get('articles', [])
    
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def export_to_csv(df):
    """Export DataFrame to CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="news_sentiment_analysis.csv">Download CSV</a>'
    return href

def plot_sentiment_trend(df):
    """Plot sentiment trend over time"""
    df['date'] = pd.to_datetime(df['Published'])
    daily_sentiment = df.groupby('date')['Sentiment Score'].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_sentiment['date'],
        y=daily_sentiment['Sentiment Score'],
        mode='lines+markers',
        name='Average Sentiment'
    ))
    fig.update_layout(
        title='Sentiment Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Sentiment Score'
    )
    return fig

def main():
    # Initialize NLTK components
    sia, stop_words = initialize_nltk()
    if sia is None or stop_words is None:
        st.error("Failed to initialize NLTK components. Please try refreshing the page.")
        return
        
    init_session_state()
    
    st.title("📰 Advanced News Sentiment Analyzer")
    st.markdown("Analyze sentiment trends and patterns in news headlines across different sources and topics.")

    # Sidebar Configuration
    st.sidebar.header("Settings")
    
    # API Key input with validation
    api_key = st.sidebar.text_input("Enter NewsAPI Key", type="password")
    if not api_key:
        st.warning("Please enter your NewsAPI key to begin analysis.")
        return

    # Analysis Options
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Real-time Analysis", "Historical Comparison"]
    )

    # Common settings
    categories = ['general', 'business', 'technology', 'science', 'health', 'entertainment', 'sports']
    selected_category = st.sidebar.selectbox("Category", categories)
    
    countries = {
        'United States': 'us', 'United Kingdom': 'gb', 'India': 'in',
        'Canada': 'ca', 'Australia': 'au', 'Germany': 'de',
        'France': 'fr', 'Japan': 'jp', 'Brazil': 'br'
    }
    selected_country = st.sidebar.selectbox("Country", list(countries.keys()))
    
    search_method = st.sidebar.radio(
        "Search Method",
        ["Top Headlines", "Everything Search"]
    )
    
    num_headlines = st.sidebar.slider("Number of Headlines", 5, 100, 20)
    
    if search_method == "Everything Search":
        date_range = st.sidebar.slider("Date Range (days)", 1, 30, 7)
    else:
        date_range = None

    try:
        # Fetch and analyze news
        with st.spinner("Fetching and analyzing news..."):
            articles = fetch_news(
                api_key, selected_category, countries[selected_country],
                num_headlines, search_method, date_range
            )

            if not articles:
                st.warning("No articles found. Try adjusting your search parameters.")
                return

            # Process articles
            news_data = []
            all_text = ""  # For wordcloud
            
            for article in articles:
                if article.get('title'):
                    headline = article['title']
                    sentiment_score = sia.polarity_scores(headline)
                    keywords = extract_keywords(headline, 3)
                    
                    news_data.append({
                        'Headline': headline,
                        'Source': article['source']['name'],
                        'Published': article.get('publishedAt', 'N/A')[:10],
                        'Sentiment Score': sentiment_score['compound'],
                        'Sentiment': get_sentiment_label(sentiment_score['compound']),
                        'Keywords': ', '.join(keywords.keys()),
                        'URL': article['url']
                    })
                    
                    all_text += f" {headline}"

            df = pd.DataFrame(news_data)
            
            # Update historical data
            if analysis_mode == "Historical Comparison":
                st.session_state.historical_data = pd.concat([st.session_state.historical_data, df])

        # Display Analysis Results
        st.header("Analysis Results")
        
        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Articles", len(df))
        with col2:
            avg_sentiment = df['Sentiment Score'].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        with col3:
            positive_pct = (df['Sentiment Score'] > 0.05).mean() * 100
            st.metric("Positive Headlines", f"{positive_pct:.1f}%")
        with col4:
            negative_pct = (df['Sentiment Score'] < -0.05).mean() * 100
            st.metric("Negative Headlines", f"{negative_pct:.1f}%")

        # Visualizations
        st.subheader("Sentiment Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment Distribution
            fig_pie = px.pie(
                df,
                names='Sentiment',
                title='Sentiment Distribution',
                color='Sentiment',
                color_discrete_map={
                    'Positive 📈': '#4CAF50',
                    'Neutral ➖': '#9E9E9E',
                    'Negative 📉': '#F44336'
                }
            )
            st.plotly_chart(fig_pie)
            
        with col2:
            # Source Analysis
            source_sentiment = df.groupby('Source')['Sentiment Score'].agg(['mean', 'count']).reset_index()
            source_sentiment = source_sentiment[source_sentiment['count'] > 1]  # Filter for sources with multiple articles
            
            fig_bar = px.bar(
                source_sentiment,
                x='Source',
                y='mean',
                title='Average Sentiment by News Source',
                color='mean',
                color_continuous_scale=['#F44336', '#9E9E9E', '#4CAF50']
            )
            st.plotly_chart(fig_bar)

        # Sentiment Trend
        st.subheader("Sentiment Trends")
        trend_fig = plot_sentiment_trend(df)
        st.plotly_chart(trend_fig)

        # Word Cloud
        st.subheader("Key Topics")
        wordcloud = create_wordcloud(all_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Detailed Headlines
        st.subheader("Analyzed Headlines")
        for _, row in df.iterrows():
            sentiment_class = get_sentiment_color(row['Sentiment Score'])
            st.markdown(f"""
            <div class="headline-card">
                <h4>{row['Headline']}</h4>
                <p>
                    <span class="source-tag">{row['Source']}</span>
                    <span class="date-tag">{row['Published']}</span>
                    <span class="{sentiment_class}">{row['Sentiment']}</span>
                </p>
                <p>Keywords: {row['Keywords']}</p>
                <a href="{row['URL']}" target="_blank">Read More</a>
            </div>
            """, unsafe_allow_html=True)

        # Export Options
        st.subheader("Export Data")
        st.markdown(export_to_csv(df), unsafe_allow_html=True)

        # Historical Comparison
        if analysis_mode == "Historical Comparison" and not st.session_state.historical_data.empty:
            st.subheader("Historical Comparison")
            historical_stats = st.session_state.historical_data.groupby(
                pd.to_datetime(st.session_state.historical_data['Published']).dt.date
            )['Sentiment Score'].agg(['mean', 'count']).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_stats['Published'],
                y=historical_stats['mean'],
                mode='lines+markers',
                name='Average Sentiment'
            ))
            fig.update_layout(title='Historical Sentiment Trend')
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your API key and try again.")

if __name__ == "__main__":
    main()