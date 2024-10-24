# Advanced News Sentiment Analyzer

A powerful web application built with Streamlit that analyzes sentiment trends and patterns in news headlines across different sources and topics. The application provides real-time analysis and historical comparison of news sentiment using natural language processing.

## Features

- Real-time sentiment analysis of news headlines
- Historical sentiment comparison
- Multiple visualization types:
  - Sentiment distribution pie chart
  - Source-wise sentiment analysis
  - Sentiment trends over time
  - Interactive word cloud of key topics
- Customizable news sources and categories
- Support for multiple countries
- Detailed headline analysis with keywords
- Data export functionality
- Dark mode support

## Requirements

- Python 3.8+
- NewsAPI Key (Get it from [https://newsapi.org](https://newsapi.org))

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/news-sentiment-analyzer.git
cd news-sentiment-analyzer
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Download required NLTK data:

```python
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

1. Get your API key from [NewsAPI](https://newsapi.org)

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Enter your NewsAPI key in the sidebar when prompted

4. Configure your analysis preferences:
   - Select analysis mode (Real-time or Historical)
   - Choose news category
   - Select country
   - Set search method and parameters
   - Adjust the number of headlines to analyze

## Configuration Options

### Categories

- General
- Business
- Technology
- Science
- Health
- Entertainment
- Sports

### Supported Countries

- United States
- United Kingdom
- India
- Canada
- Australia
- Germany
- France
- Japan
- Brazil

### Analysis Modes

1. **Real-time Analysis**: Analyze current news headlines
2. **Historical Comparison**: Track sentiment trends over time

## Features in Detail

### Sentiment Analysis

- Utilizes NLTK's VADER sentiment analyzer
- Provides compound sentiment scores
- Categorizes headlines as Positive, Negative, or Neutral

### Visualizations

1. **Sentiment Distribution**: Pie chart showing the distribution of positive, negative, and neutral headlines
2. **Source Analysis**: Bar chart displaying average sentiment by news source
3. **Sentiment Trends**: Line chart tracking sentiment changes over time
4. **Word Cloud**: Visual representation of frequently occurring terms

### Data Export

- Export analysis results to CSV
- Includes all metrics and sentiment scores
- Compatible with spreadsheet software

## Technical Details

- Built with Streamlit for the web interface
- Uses NLTK for natural language processing
- Implements NewsAPI for data retrieval
- Visualizations created using Plotly and Matplotlib
- Responsive design with dark mode support

## Error Handling

The application includes robust error handling for:

- Invalid API keys
- Network connection issues
- Rate limiting
- Invalid search parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [NewsAPI](https://newsapi.org) for providing the news data
- [NLTK](https://www.nltk.org/) for sentiment analysis tools
- [Streamlit](https://streamlit.io) for the web framework

## Support

For support, please open an issue in the GitHub repository or contact [your contact information].
