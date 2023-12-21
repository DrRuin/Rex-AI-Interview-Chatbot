from openai import OpenAI
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat/text')
def text_chat():
    return render_template('text_chat.html')

@app.route('/chat/speech')
def speech_chat():
    return render_template('speech_chat.html')

client = OpenAI(api_key="Key")
personality = """
Task: You Are RexAI, Assume the role of a Senior Data Scientist at Google, specializing in recent technologies like LLMs and long-chain algorithms. You were assigned to follow these rules:

Rules to Follow:
1. Maintain a strict focus on the data science interview; do not deviate from the topic.
2. If the user goes off-topic, gently remind them to stay focused on the interview.
3. Create and present up-to-date questions relevant to the field of data science.

Interview Structure:
The interview will consist of three rounds:
- Round 1: This round will include 10 challenging multiple-choice questions. You are to ask these questions to the user.
- Round 2: This round will consist of 10 short-answer questions. You will ask these questions to the user.
- Round 3: This round will involve 10 detailed, long-form questions. You will ask these in-depth questions to the user.

Evaluation Criteria:
- Round 1: Each question is worth 1 point. Award 1 point for each correct answer and 0 points for incorrect answers. If the user scores 6 points or fewer, end the interview with the message, "Sorry, we won't be moving forward with you."
- Round 2: Each question is worth 2 points. Award 2 points for each correct answer and 0 points for incorrect answers. If the user scores 10 points or fewer, end the interview with the message, "Sorry, we won't be moving forward with you."
- Round 3: Each question is worth 3 points. Award 3 points for each correct answer and 0 points for incorrect answers. If the user scores 15 points or fewer, end the interview with the message, "Sorry, we won't be moving forward with you."

- If the user fails in any round, terminate the session and start again from Round 1.
- If the user successfully passes Round 1, proceed to Round 2. However, if the user fails in Round 2, terminate the session and restart from Round 1.
- If the user passes both Round 1 and Round 2 but fails in Round 3, terminate the session and begin again from Round 1. 

Please adhere strictly to these guidelines!
"""

messages = [{"role": "system", "content": f"{personality}"}]

def generate_audio(text, filename="speech_output.mp3"):
    speech_file_path = Path(__file__).parent / filename
    max_length = 4096

    text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    for chunk in text_chunks:
        response = client.audio.speech.create(model="tts-1", voice="nova", input=chunk)
        response.stream_to_file(speech_file_path)

    return speech_file_path

def generate_text():
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    bot_response = response.choices[0].message.content
    messages.append({"role": "assistant", "content": bot_response})
    return bot_response

def save_history_to_json(history, file_path="history.json"):
    with open(file_path, "w") as file:
        json.dump(history, file, indent=4)

def load_history_from_json(file_path="history.json"):
    with open(file_path, "r") as file:
        history = json.load(file)
    return history

@app.route('/get_last_messages')
def get_last_messages():
    if messages:
        user_message = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), None)
        bot_reply = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'assistant'), None)
        return jsonify({'user_message': user_message, 'bot_reply': bot_reply})
    return jsonify({'user_message': '', 'bot_reply': ''})


@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.json['message']
    messages.append({"role": "user", "content": user_input})
    bot_response = generate_text()
    messages.append({"role": "assistant", "content": bot_response})
    save_history_to_json(messages, "history.json")
    return jsonify({'reply': bot_response})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files['audio']
    audio_file_path = Path(__file__).parent / "temp_voice.mp3"
    audio_file.save(audio_file_path)

    transcript = client.audio.transcriptions.create(model="whisper-1", file=open(audio_file_path, "rb")).text
    messages.append({"role": "user", "content": transcript})

    bot_response = generate_text()
    audio_response_path = generate_audio(bot_response)

    messages.append({"role": "assistant", "content": bot_response})
    save_history_to_json(messages, "history.json")

    return send_file(audio_response_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
