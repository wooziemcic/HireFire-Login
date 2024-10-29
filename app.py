from flask import Flask, redirect, url_for, render_template, request, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from transformers import pipeline
import traceback
import os
import spacy
import fitz
import requests
import speech_recognition as sr
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import io
import string
import nltk
import numpy as np
import cv2
from deepface import DeepFace
import logging


# Load environment variables
load_dotenv(override=True)

# Load environment variables
load_dotenv(override=True)

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///interviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'
nlp = spacy.load('en_core_web_sm')


# Initialize SQLAlchemy and LoginManager
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_github'

# Initialize the FLAN-T5 model and speech recognizer
generator = pipeline('text2text-generation', model='google/flan-t5-large')
recognizer = sr.Recognizer()
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
nltk.download('stopwords')

# Database model for storing interview data
class Interview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_description = db.Column(db.Text, nullable=False)
    questions = db.Column(db.Text, nullable=False)
    transcription = db.Column(db.Text, nullable=True)
    score = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(10), nullable=True)

# Create the database tables
with app.app_context():
    db.create_all()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)

    def __repr__(self):
        return f'<User {self.username}>'


# Debugging: Check if the environment variables are loaded
print("DEBUG: GITHUB_CLIENT_ID =", os.getenv('GITHUB_CLIENT_ID'), flush=True)
print("DEBUG: GITHUB_CLIENT_SECRET =", os.getenv('GITHUB_CLIENT_SECRET'), flush=True)

# GitHub OAuth Setup with debug prints
oauth = OAuth(app)

github = oauth.register(
    name='github',
    client_id=os.getenv('GITHUB_CLIENT_ID'),
    client_secret=os.getenv('GITHUB_CLIENT_SECRET'),
    authorize_url='https://github.com/login/oauth/authorize',
    token_url='https://github.com/login/oauth/access_token',
    userinfo_url='https://api.github.com/user',
    client_kwargs={
        'scope': 'user:email',
        'token_endpoint_auth_method': 'client_secret_post',
    }
)

# Verify that the token URL is registered
print(f"DEBUG: OAuth Token URL = https://github.com/login/oauth/access_token", flush=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/login/github')
def login_github():
    redirect_uri = url_for('github_callback', _external=True)
    print(f"DEBUG: Redirect URI = {redirect_uri}", flush=True)
    return github.authorize_redirect(redirect_uri)

import requests

@app.route('/login/callback')
def github_callback():
    try:
        # Log the request args for debugging
        print(f"DEBUG: Request Args = {request.args}", flush=True)

        # Extract the code from the request
        code = request.args.get('code')
        redirect_uri = url_for('github_callback', _external=True)
        print(f"DEBUG: Code = {code}, Redirect URI = {redirect_uri}", flush=True)

        # Manually exchange the authorization code for an access token
        response = requests.post(
            'https://github.com/login/oauth/access_token',
            data={
                'client_id': os.getenv('GITHUB_CLIENT_ID'),
                'client_secret': os.getenv('GITHUB_CLIENT_SECRET'),
                'code': code,
                'redirect_uri': redirect_uri,
            },
            headers={'Accept': 'application/json'}
        )

        # Log the token exchange response
        token = response.json()
        print(f"DEBUG: Token Response = {token}", flush=True)

        if 'access_token' not in token:
            raise ValueError("Failed to retrieve access token.")

        # Use the access token to fetch user information
        user_info_response = requests.get(
            'https://api.github.com/user',
            headers={'Authorization': f"token {token['access_token']}"}
        )
        user_info = user_info_response.json()
        print(f"DEBUG: User Info = {user_info}", flush=True)

        # Handle user login or creation
        username = user_info['login']
        email = user_info.get('email', f"{username}@github.com")

        user = User.query.filter_by(username=username).first()
        if not user:
            user = User(username=username, email=email)
            db.session.add(user)
            db.session.commit()

        login_user(user)
        return redirect(url_for('home'))

    except Exception as e:
        # Log any exception that occurs
        print(f"ERROR: OAuth callback failed: {traceback.format_exc()}", flush=True)
        return f"<h3>Error: {str(e)}</h3>"

@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume = request.files['resume']
        resume_text = extract_text_from_pdf(resume)

        if check_resume_fit(job_description, resume_text):
            questions = generate_questions(job_description)
            new_interview = Interview(job_description=job_description, questions="\n".join(questions))
            db.session.add(new_interview)
            db.session.commit()
            return redirect(url_for('questions', interview_id=new_interview.id))
        else:
            return "<h3>You are not eligible for this job.</h3>"

    return render_template('home.html', username=current_user.username)

@app.route('/questions/<int:interview_id>', methods=['GET'])
@login_required
def questions(interview_id):
    interview = Interview.query.get_or_404(interview_id)
    return render_template('questions.html', questions=interview.questions.split('\n'), interview_id=interview_id)

@app.route('/record_answer/<int:interview_id>', methods=['POST'])
@login_required
def record_answer(interview_id):
    try:
        # Get the video data from the form
        video_data = request.form.get('video_data', None)
        if not video_data:
            raise ValueError("No video data received. Please try again.")

        print(f"Received video data (first 100 chars): {video_data[:100]}")

        # Analyze the video for a confidence score
        confidence_score = analyze_video(video_data)
        print(f"Confidence Score: {confidence_score}")

        # Transcribe the audio from the video
        transcription = transcribe_from_video(video_data)
        if not transcription:
            print("DEBUG: No valid speech detected.")
            transcription = ""  # Set transcription to an empty string

        print(f"Transcription: {transcription}")

        # Retrieve the interview from the database
        interview = Interview.query.get(interview_id)
        if not interview:
            raise ValueError(f"Interview with ID {interview_id} not found.")

        # Calculate NLP score if there is a transcription, else assign 0
        nlp_score = score_transcription(transcription, interview.job_description) if transcription else 0.0
        print(f"NLP Score: {nlp_score}")

        # Calculate the final score
        final_score = (0.4 * confidence_score) + (0.6 * nlp_score)
        status = "Hired" if final_score >= 0.5 else "Not Hired"
        print(f"Final Score: {final_score}, Status: {status}")

        # Update the interview record with the results
        interview.transcription = transcription
        interview.score = final_score
        interview.status = status
        db.session.commit()

        # Render the result page with the scores and status
        return render_template(
            'result.html',
            transcription=transcription,
            confidence_score=confidence_score,
            nlp_score=nlp_score,
            final_score=final_score,
            status=status
        )

    except Exception as e:
        print(f"Error occurred: {traceback.format_exc()}")
        return f"<h3>Error: {str(e)}</h3>"


def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def score_transcription(transcription, job_description):
    """Calculate the similarity score between transcription and job description."""
    transcription_clean = preprocess_text(transcription)
    job_description_clean = preprocess_text(job_description)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_description_clean, transcription_clean])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    similarity_score = similarity_matrix[0, 1]  # Get similarity between two inputs
    print(f"Similarity Score: {similarity_score}")
    return similarity_score

def extract_text_from_pdf(pdf_file):
    pdf = fitz.open(stream=pdf_file.read(), filetype='pdf')
    text = "".join([page.get_text() for page in pdf])
    pdf.close()
    return text

def check_resume_fit(job_description, resume_text):
    job_doc = nlp(job_description)
    resume_doc = nlp(resume_text)
    return job_doc.similarity(resume_doc) >= 0.8

def generate_questions(description):
    questions = []
    for q_type in ['technical', 'non-technical']:
        prompt = f"Generate one {q_type} question for: {description}"
        result = generator(prompt, max_new_tokens=50)[0]['generated_text']
        questions.append(result.strip())
    return questions

def analyze_video(video_data):
    try:
        # Decode the base64 video data
        video_bytes = base64.b64decode(video_data)
        video_buffer = io.BytesIO(video_bytes)

        # Write the video to a temporary file
        with open("temp_video.webm", "wb") as f:
            f.write(video_buffer.read())

        # Open the video file using OpenCV
        cap = cv2.VideoCapture("temp_video.webm")
        if not cap.isOpened():
            raise ValueError("Failed to open video for analysis.")

        frame_count = 0
        emotion_scores = {"happy": 0, "neutral": 0, "angry": 0, "surprise": 0}

        # Process each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()

            # Check if the frame was read successfully
            if not ret or frame is None:
                print("DEBUG: Invalid frame or end of video.")
                break  # Exit loop if no more frames or invalid frame found

            # Ensure the frame has a valid size
            if frame.size == 0:
                print("DEBUG: Empty frame encountered, skipping.")
                continue

            # Optional: Resize frame for consistency if needed (e.g., to 640x480)
            frame = cv2.resize(frame, (640, 480))

            try:
                # Analyze the frame using DeepFace for emotion detection
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)[0]
                emotion = result.get('dominant_emotion', 'neutral')
                if emotion in emotion_scores:
                    emotion_scores[emotion] += 1
            except Exception as e:
                print(f"Error analyzing frame: {e}")

            frame_count += 1

        # Release video capture resources
        cap.release()
        cv2.destroyAllWindows()

        if frame_count == 0:
            raise ValueError("No valid frames found in the video.")

        # Normalize emotion scores by dividing by the total frame count
        for emotion in emotion_scores:
            emotion_scores[emotion] /= max(frame_count, 1)

        # Calculate the final confidence score using the weighted formula
        return (0.6 * emotion_scores["happy"]) + (0.4 * emotion_scores["neutral"])

    except Exception as e:
        print(f"Error analyzing video: {traceback.format_exc()}")
        return 0.0

def transcribe_from_video(video_data):
    try:
        # Decode and extract audio from the video
        video_bytes = base64.b64decode(video_data)
        audio = AudioSegment.from_file(io.BytesIO(video_bytes), format="webm")

        # Save the audio for debugging purposes
        with open("extracted_audio.wav", "wb") as audio_file:
            audio.export(audio_file, format="wav")

        # Check audio loudness and log it
        print(f"DEBUG: Audio loudness (dBFS): {audio.dBFS}")
        if audio.dBFS < -60:
            print("DEBUG: No valid speech detected (too quiet or silent).")
            return ""  # Return an empty transcription if the audio is too quiet

        # Convert the audio to WAV in-memory for speech recognition
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        # Use the recognizer to transcribe the audio
        with sr.AudioFile(wav_buffer) as source:
            audio_content = recognizer.record(source, duration=30)  # Limit duration
            try:
                transcription = recognizer.recognize_google(audio_content)
                print(f"DEBUG: Transcription: {transcription}")
                return transcription
            except sr.UnknownValueError:
                print("DEBUG: No speech recognized.")
                return ""  # Return empty string if no valid speech is detected
            except sr.RequestError as e:
                print(f"ERROR: Request error with Google API: {e}")
                return ""  # Return empty string if there is an API error

    except Exception as e:
        print(f"Error during transcription: {traceback.format_exc()}")
        return ""  # Return empty string on any other errors

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login_github'))

if __name__ == "__main__":
    app.run(debug=True)
