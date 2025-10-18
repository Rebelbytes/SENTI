# SENTI
NLP_miniprj

# YouTube Sentiment Analysis Web App

This is a Flask-based web application that performs sentiment analysis on YouTube video comments. It fetches comments from a specified YouTube video, analyzes their sentiment using a pre-trained Naive Bayes model, and displays the results including sentiment percentages, a pie chart, and top comments categorized by sentiment.

## Features

- Fetch YouTube video comments using the YouTube Data API v3
- Preprocess and clean text data
- Perform sentiment analysis (Positive, Negative, Neutral) using TF-IDF vectorization and Multinomial Naive Bayes
- Generate a pie chart visualizing sentiment distribution
- Display top 10 comments for each sentiment category
- Responsive web interface using Bootstrap

## Installation

1. Clone or download this repository to your local machine.

2. Install the required Python packages:
   ```
   pip install flask google-api-python-client scikit-learn datasets pandas numpy matplotlib
   ```

3. Obtain a YouTube Data API v3 key from the Google Cloud Console and replace the placeholder in `app.py`:
   - Find the line: `developerKey='AIzaSyCcBrt-UsC9soIxO-y5wu3z8xGzyuu2rIE'`
   - Replace with your own API key.

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. Enter a YouTube video URL in the input field and click "Analyze".

4. View the sentiment analysis results, including the pie chart and categorized comments.

## Project Structure

- `app.py`: Main Flask application with sentiment analysis logic
- `templates/index.html`: Home page for entering video URL
- `templates/senti_result.html`: Results page displaying analysis
- `static/sentiment_pie_chart.png`: Generated pie chart image

## Dependencies

- Flask
- google-api-python-client
- scikit-learn
- datasets (Hugging Face)
- pandas
- numpy
- matplotlib

## Notes

- The model is trained on a preprocessed sentiment analysis dataset from Hugging Face.
- Ensure you have a valid YouTube API key to fetch comments.
- The application runs in debug mode by default.

## License

This project is for educational purposes. Please ensure compliance with YouTube's API terms of service.

