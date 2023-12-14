from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import re

nltk.download('vader_lexicon')
app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def main():
    overall_sentiment = None
    video_details = None

    if request.method == "POST":
        try:
            youtube_url = request.form.get("youtube_url")
            video_id = extract_video_id(youtube_url)
            video_details = get_video_details(video_id)
            title_sentiment = analyze_sentiment(video_details["title"])
            description_sentiment = analyze_sentiment(video_details["description"])
            overall_sentiment = "Good" if title_sentiment == "Positive" and description_sentiment == "Positive" else "Bad"
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            overall_sentiment = "Good" 
        
    return render_template('home.html', video_details=video_details, overall_sentiment=overall_sentiment)

def extract_video_id(url):
    video_id_match = re.search(r"(?<=v=|\/videos\/|embed\/|youtu.be\/|\/v\/|\/e\/|watch\?v=|%2Fvideos%2F|embed%\2F|youtu.be%2F|%2Fv%2F)([a-zA-Z0-9_-]{11})", url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        return None

def get_video_details(video_id):
    api_key = "YOUR_YOUTUBE_API_KEY" 
    url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet"
    response = requests.get(url)
    data = response.json()
    
    if "items" in data and data["items"]:
        snippet = data["items"][0]["snippet"]
        title = snippet["title"]
        description = snippet["description"]
        return {"title": title, "description": description}
    else:
        return {"title": "Video not found", "description": "Video details not available"}

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)
    if score["compound"] >= 0.05:
        return "Positive"
    elif score["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

if __name__ == '__main__':
    app.run(debug=True)
