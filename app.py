import json
import os
import sys
from openai import OpenAI

from pytube import YouTube
import argparse
import requests
from dotenv import load_dotenv

from database import init_db, Session
from models import YouTubeAudioData
import datetime

import streamlit as st

#GPT_MODEL='gpt-3.5-turbo'
GPT_MODEL='gpt-4-1106-preview'


load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def download_audio_from_youtube_to_file(url):
    yt = YouTube(url)
    out_file = yt.streams.get_audio_only().download("./data/audio_files/")
    output_mp3 = os.path.splitext(out_file)[0] + '.mp3'
    os.rename(out_file, output_mp3)
    print(f'Audio file downloaded to: {output_mp3}')

    return output_mp3


def transcribe_audio(audio_file):
    with open(audio_file, 'rb') as file:
        transcript = client.audio.transcriptions.create(model='whisper-1', 
        file=file, response_format="text")
    print(f"Transcript from the audio:\n\n{transcript}\n")

    return transcript

def summarize_transcript(transcript, model):
    completion = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that "
            "summarizes transcriptions from audio files as though you are the speaker."},
        {"role": "user", "content": (f"First summarize the most important topics "
            f"from the following transcript as though you are the speaker, emphasizing the most critical ideas "
            f"for your telegram channel. Do not reference that you're summarizing a transcript: {transcript}")
        },
    ])

    return completion.choices[0].message.content


# Define your handler functions
def handle_pending(audio_record):
    # Logic for handling pending status
    # Download audio, update status, etc.

    audio_file = download_audio_from_youtube_to_file(audio_record.url)
    audio_record.audio_file_path = audio_file
    audio_record.status = 'downloaded'

def handle_downloaded(audio_record):
    # Logic for handling downloaded status
    # Transcribe audio, update status, etc.

    audio_file_path = audio_record.audio_file_path
    transcript = transcribe_audio(audio_file_path)

    # Save the transcript to a file
    transcript_file = f"./data/transcripts/{os.path.basename(audio_file_path)}.txt"
    with open(transcript_file, 'w') as f:
        f.write(transcript)

    audio_record.transcript_file_path = transcript_file
    audio_record.status = "transcribed"

def handle_transcribed(audio_record):
    # Logic for handling transcribed status
    # Summarize transcript, update status, etc.
    with open(audio_record.transcript_file_path, 'r') as f: 
        transcript = f.read()

    model=GPT_MODEL
    summary = summarize_transcript(transcript, model)

    # Save the summary to a file
    summary_file = f"./data/summaries/{os.path.basename(audio_record.audio_file_path)}.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)

    audio_record.transcript_summary_path = summary_file
    audio_record.summary_model_used = model
    audio_record.status = "summarized"

    pass

def handle_summarized(audio_record):
    # assume you're getting called to be force resummarized
    handle_transcribed(audio_record)

def handle_error(audio_record):
    # Logic for handling error status
    # Error recovery or notification
    raise SystemError("There was an error for some reason. Time to debug")

def handle_youtube_url(url, force_resummarization=False):
    # Define a map from status to handler function
    status_handlers = {
        'pending': handle_pending,
        'downloaded': handle_downloaded,
        'transcribed': handle_transcribed,
        'summarized': handle_summarized,
        'error': handle_error
    }

    session = Session()
    audio_data = session.query(YouTubeAudioData).filter_by(url=url).first()

    if not audio_data:
        # Insert a new record with status 'pending'
        audio_data = YouTubeAudioData(
            url=url,
            title="Placeholder Title",
            description="Placeholder description.",
            status='pending',
            index_date=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )
        session.add(audio_data)
        print(f"New URL: {url} inserted, ready for processing.")

    status_before = audio_data.status
    # Loop through successive states until an error occurs or we make it to the `summarized` state
    while audio_data.status != 'error' and audio_data.status != 'summarized':
        
        # Get the handler based on the status
        handler = status_handlers.get(audio_data.status, lambda x: print(f"Unhandled status: {audio_data.status}"))
        # Call the handler function
        handler(audio_data)
        # Refresh the audio_data object to get the updated status
        session.commit()
        session.refresh(audio_data)

    session.commit()

    session.refresh(audio_data)
    if force_resummarization and status_before == audio_data.status == 'summarized': 
        handle_transcribed(audio_data)
    else: 
        assert audio_data.status == 'summarized'

    audio_data.load_summary()
    return audio_data.summary


def post_to_telegram(summary, audio=None):
    TELEGRAM_BOT_API_KEY = os.environ.get('TEST_CHANNEL_9000_BOT_API_KEY')
    MY_CHANNEL_NAME = os.environ.get("TEST_CHANNEL_ID")

    response = requests.get(f'https://api.telegram.org/bot{TELEGRAM_BOT_API_KEY}/sendMessage', {
        'chat_id': MY_CHANNEL_NAME,
        'text': f"{summary}\n\nDisclaimer: the above summary was generated by a large language model. Take it with a grain of salt."
    })

    if not response.status_code == 200:
        print(response.text)

    if audio == None: return

    with open(audio, 'rb') as audio_file:
        response = requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_BOT_API_KEY}/sendAudio',
            data={
                'chat_id': MY_CHANNEL_NAME,
                'title': 'Audio Summary',
                'caption': 'Here is the audio file related to the summary posted earlier.'
            },
            files={
                'audio': audio_file
            }
        )

    if not response.status_code == 200:
        print(response.text)


def main():

    # initialize the local database for tracking audio file, transcripts, summaries, etc.
    # schemas are defined in `models.py`
    init_db()

    st.title("YouTube Audio Transcriber and Summarizer")

    input_type = st.radio("Choose input type", ("YouTube URL", "Upload an audio file"))

    url = audio_file = None
    if input_type == "YouTube URL":
        url = st.text_input("Enter a YouTube URL:")
    else:
        audio_file = st.file_uploader("Upload an audio file:", type=['mp3', 'wav'])

    disable_telegram = st.checkbox("Disable posting to Telegram", value=True)

    force_resummarization = st.checkbox("Force resummarization", value=False)

    if st.button("Process"):
        if url:
            summary = handle_youtube_url(url, force_resummarization)
            if not disable_telegram:
                post_to_telegram(summary)
            st.markdown(f"**Summary for the YouTube video at {url}:**")
            st.text_area("Summary", summary, height=250)
        elif audio_file:
            # Save the uploaded audio file to a temporary location
            audio_path = f"./data/audio_files/{audio_file.name}"
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())
            raise NotImplemented("this is old dead code. Need to reimplement this path")
            if not disable_telegram:
                post_to_telegram(summary, audio_path)
            st.markdown(f"**Summary for the uploaded audio file {audio_file.name}:**")
            st.text_area("Summary", summary, height=250)
        else:
            st.warning("Please enter a URL or upload an audio file.")

if __name__ == '__main__':
    main()