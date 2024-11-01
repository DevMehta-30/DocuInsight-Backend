from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import fitz  
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
import whisper
import yt_dlp
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from random import shuffle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://docu-insight-frontend.vercel.app", "http://localhost:5173"]}})


# Define the path for temporary files
TEMP_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Helper Classes and Functions
class SimpleTokenizer:
    """Tokenizer for splitting text into sentences and words."""
    def to_sentences(self, text):
        return text.split('. ')
    
    def to_words(self, text):
        return re.findall(r'\w+', text.lower())

def summarize_text(text, sentence_count=10):
    """Summarizes text into a specified number of sentences."""
    parser = PlaintextParser.from_string(text, SimpleTokenizer())
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return ' '.join([str(sentence) for sentence in summary])

def extract_text_from_image(image_path):
    """Extracts text from an image using pytesseract."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of a PDF using PyMuPDF."""
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    pdf_document.close()
    return text

def download_youtube_audio(url):
    """Downloads the audio of a YouTube video."""
    audio_path = os.path.join(TEMP_DIR, 'audio.mp4')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return audio_path

def transcribe_audio(audio_file):
    """Transcribes audio using Whisper."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result['text']

def extract_nouns_verbs(sentence):
    """Extracts contextually important nouns and verbs from a sentence using POS tagging."""
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    # Define a set of stop words and irrelevant tags
    stop_words = set(stopwords.words("english"))
    keyword_tags = {"NN", "NNS", "NNP", "VB", "VBP", "VBZ", "VBD", "VBG", "VBN"}
    
    # Extract only meaningful nouns and verbs
    keywords = [word for word, pos in pos_tags if pos in keyword_tags and word.lower() not in stop_words]
    return keywords

def generate_question(sentence):
    """Generates a meaningful question by replacing a specific keyword in the sentence with a blank."""
    keywords = extract_nouns_verbs(sentence)
    
    if not keywords:
        return None, None  # Skip if no meaningful keyword is found
    
    # Select a contextually significant keyword
    for keyword in keywords:
        if len(keyword) > 3:  # Avoid trivial short words as blanks
            question = sentence.replace(keyword, '______', 1)
            if question.endswith('.'):
                question = question[:-1] + '?'  # Ensure it ends with a question mark
            return question, keyword
    
    return None, None

def generate_quiz(text):
    """Generates a quiz from the given text, skipping metadata and focusing on contextually meaningful sentences."""
    sentences = sent_tokenize(text)
    questions_and_answers = []

    for sentence in sentences:
        question, answer = generate_question(sentence)
        if question:
            questions_and_answers.append((question, answer))

    return questions_and_answers

# Route Definitions
@app.route('/process_files', methods=['POST', 'OPTIONS'])
def process_files_api():
    """Processes uploaded files (PDF or images) and returns extracted text and summary."""
    if request.method == "OPTIONS":  # Handle the CORS preflight request
        return build_cors_preflight_response()

    files = request.files.getlist("files")
    extracted_text = ""

    for file in files:
        file_path = os.path.join(TEMP_DIR, file.filename)
        file.save(file_path)

        try:
            if file.filename.endswith(('.png', '.jpg', '.jpeg')):
                extracted_text += extract_text_from_image(file_path)
            elif file.filename.endswith('.pdf'):
                extracted_text += extract_text_from_pdf(file_path)
        except Exception as e:
            return jsonify({"error": f"Failed to process {file.filename}: {str(e)}"}), 500
        finally:
            os.remove(file_path)    

    if extracted_text:
        return jsonify({"text": extracted_text})
    else:
        return jsonify({"error": "No text extracted from the provided files."})

@app.route('/summarize_youtube', methods=['POST', 'OPTIONS'])
def summarize_youtube_api():
    """Summarizes text from a YouTube video by downloading, transcribing, and summarizing its audio."""
    if request.method == "OPTIONS":  # Handle the CORS preflight request
        return build_cors_preflight_response()

    data = request.get_json()
    url = data.get("url")
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        audio_file = download_youtube_audio(url)
        transcript = transcribe_audio(audio_file)
    except Exception as e:
        return jsonify({"error": f"Failed to summarize YouTube video: {str(e)}"}), 500
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
    return jsonify({"transcript": transcript})

@app.route('/transcribe_audio', methods=['POST', 'OPTIONS'])
def transcribe_audio_api():
    """Transcribes and summarizes uploaded audio files."""
    if request.method == "OPTIONS":  # Handle the CORS preflight request
        return build_cors_preflight_response()

    audio_file = request.files.get("audio_file")
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_path = os.path.join(TEMP_DIR, "uploaded_audio.mp4")
    audio_file.save(audio_path)
    
    try:
        transcript = transcribe_audio(audio_path)
    except Exception as e:
        return jsonify({"error": f"Failed to transcribe audio: {str(e)}"}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
    return jsonify({"transcript": transcript})

from flask import make_response

@app.route('/summarize', methods=['POST'])
def summarize_api():
    text = request.json.get("content")
    if not text:
        return jsonify({"error": "No content provided"}), 400
    try:
        summary = summarize_text(text)
        response = make_response(jsonify({"summary": summary}))
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin")
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response
    except Exception as e:
        return jsonify({"error": f"Failed to summarize: {str(e)}"}), 500

@app.route('/generate_quiz', methods=['POST', 'OPTIONS'])
def generate_quiz_api():
    """Generates a quiz with meaningful questions from the provided text, with questions shuffled."""
    if request.method == "OPTIONS":  # Handle the CORS preflight request
        return build_cors_preflight_response()
    data = request.get_json()
    text = data.get("text")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        quiz = generate_quiz(text)
        shuffle(quiz)  # Shuffle the order of the questions
        limited_quiz = quiz[:10]  # Limit to 10 questions after shuffling
        questions = [{"question": q, "answer": a} for q, a in limited_quiz]
        return jsonify({"quiz": questions})
    except Exception as e:
        return jsonify({"error": f"Failed to generate quiz: {str(e)}"}), 500

# Helper function for handling CORS
def build_cors_preflight_response():
    origin = request.headers.get("Origin")
    if origin in ["https://docu-insight-frontend.vercel.app", "http://localhost:5173"]:
        response = jsonify()
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response, 200
    else:
        return jsonify({"error": "CORS not allowed"}), 403

if __name__ == '__main__':
    app.run(debug=True)
