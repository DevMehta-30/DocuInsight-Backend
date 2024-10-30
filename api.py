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
from random import choice, sample
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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

# Routes
@app.route('/process_files', methods=['POST'])
def process_files_api():
    """Processes uploaded files (PDF or images) and returns extracted text and summary."""
    files = request.files.getlist("files")
    extracted_text = ""
    
    for file in files:
        # Save file temporarily in TEMP_DIR
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

@app.route('/summarize_youtube', methods=['POST'])
def summarize_youtube_api():
    """Summarizes text from a YouTube video by downloading, transcribing, and summarizing its audio."""
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

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio_api():
    """Transcribes and summarizes uploaded audio files."""
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

@app.route('/summarize',methods=['POST'])
def summarize_api():
    """Summarize the text."""
    text= request.json.get("content")
    if not text:
        return jsonify({"error": "No content provided"}), 400

    try:
        summarize = summarize_text(text)
    except Exception as e:
        return jsonify({"error": f"Failed to summarize: {str(e)}"}), 500

    return jsonify({"summary":summarize})

if __name__ == '__main__':
    app.run(debug=True)
