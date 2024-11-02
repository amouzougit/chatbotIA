from flask import Flask, request, render_template, jsonify
from main import ChatBot
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)

# Allowed file extensions and max content size
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize the chatbot
bot = ChatBot()

@app.route('/')
def home():
    return render_template('bot_1.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_input']
    try:
        # Get the full response from the chatbot
        full_response = bot.ask_question(user_message)

        # Extract only the answer part using regex
        match = re.search(r"Answer:\s*(.*)", full_response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
        else:
            answer = full_response  # Fallback if no "Answer:" keyword is found

        # Return only the answer as the response
        return jsonify({'response': answer})

    except Exception as e:
        logging.error(f"Error in /chat route: {e}")
        return jsonify({'response': 'An error occurred while processing your request.'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part in the request.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file.'}), 400

    if file and allowed_file(file.filename):
        try:
            # Ensure the 'temp' directory exists
            if not os.path.exists('temp'):
                os.makedirs('temp')

            # Save the file to a temporary location
            filepath = os.path.join('temp', file.filename)
            file.save(filepath)

            # Process the file and update the knowledge base
            bot.process_uploaded_file(filepath)

            # Remove the temporary file
            os.remove(filepath)

            return jsonify({'status': 'success', 'message': 'File uploaded and processed successfully.'}), 200
        except Exception as e:
            logging.error(f"Error in /upload route: {e}")
            return jsonify({'status': 'error', 'message': 'Error processing the file.'}), 500
    else:
        return jsonify({'status': 'error', 'message': 'File type not allowed.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
