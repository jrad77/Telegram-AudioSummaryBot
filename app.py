import json
import os
import sys
from openai import OpenAI

import yt_dlp
import argparse
import requests
from dotenv import load_dotenv

from database import init_db, Session
from models import YouTubeAudioData, AudioData
import datetime

import streamlit as st
from pydub import AudioSegment
import shutil
import cv2
import base64

GPT_MODEL='gpt-4o'

OBSIDIAN_MARKDOWN_FILE_DESTINATION = os.path.join(os.path.expanduser("~"), "Documents", "Obsidian Vaults", "Omega", "Transcripts")


load_dotenv()

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY'),
  organization=os.getenv('OPENAI_ORGANIZATION'),
  project=os.getenv('OPENAI_PROJECT_ID'),
)


def download_audio_from_youtube_to_file(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': './data/audio_files/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        # Ensure the file extension is .mp3
        audio_file_path = ydl.prepare_filename(info_dict)
        if not audio_file_path.endswith('.mp3'):
            audio_file_path = audio_file_path.rsplit('.', 1)[0] + '.mp3'
    print(f'Audio file downloaded to: {audio_file_path}')

    return audio_file_path


def download_video_from_youtube(url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': './data/video_files/%(title)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_file_path = ydl.prepare_filename(info_dict)
    print(f'Video file downloaded to: {video_file_path}')

    return video_file_path

def process_video(video_path, fps=1):
    base64Frames = []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(video_fps / fps)

    for i in range(0, total_frames, frames_to_skip):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    return base64Frames

def transcribe_audio(audio_file):
    # Check the file size
    file_size = os.path.getsize(audio_file)
    if file_size > 25 * 1024 * 1024:  # 25MB in bytes
        # Calculate the number of chunks needed
        num_chunks = file_size // (25 * 1024 * 1024) + 1

        # Split the audio file into chunks
        audio = AudioSegment.from_file(audio_file)
        chunk_size = len(audio) // num_chunks
        chunks = [audio[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

        # Transcribe each chunk separately
        transcripts = []
        for i, chunk in enumerate(chunks):
            chunk_file = f"./data/audio_files/chunk_{i}.mp3"
            chunk.export(chunk_file, format="mp3")
            with open(chunk_file, 'rb') as file:
                print(f"Transcribing chunk {i+1}/{num_chunks}")
                transcript = client.audio.transcriptions.create(model='whisper-1', file=file, response_format="text")
            transcripts.append(transcript)

        # Combine the transcripts
        transcript = " ".join(transcripts)
    else:
        # Transcribe the entire audio file
        with open(audio_file, 'rb') as file:
            print("Transcribing audio file")
            transcript = client.audio.transcriptions.create(model='whisper-1', file=file, response_format="text")
    print(f"Transcript from the audio:\n\n{transcript}\n End of Transcript\n\n")

    return transcript

def summarize_transcript(transcript, model, base64Frames=None):
    messages = [
        {"role": "system", "content": "You are generating a video summary. Create a summary of the provided video and its transcript. Respond in Markdown. If the transcript is not in english, translate it to English."},
        {"role": "user", "content": []}
    ]

    if base64Frames:
        messages[1]["content"].extend([
            "These are the frames from the video.",
            *map(lambda x: {"type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{x}", "detail": "low"}}, 
                base64Frames)
        ])

    messages[1]["content"].append({"type": "text", "text": f"The audio transcription is: {transcript}"})

    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

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

    # Assume the audio file path is set
    audio_file_path = audio_record.audio_file_path

    # file to save transcript to
    transcript_file = f"./data/transcripts/{os.path.basename(audio_file_path)}.txt"


    print(f"Transcribing audio file: {audio_file_path}")
    transcript = transcribe_audio(audio_file_path)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(transcript_file), exist_ok=True)

    with open(transcript_file, 'w') as f:
        f.write(transcript)

    audio_record.transcript_file_path = transcript_file
    audio_record.status = "transcribed"

    # Copy transcript file to OBSIDIAN_MARKDOWN_FILE_DESTINATION
    markdown_file = os.path.join(OBSIDIAN_MARKDOWN_FILE_DESTINATION, os.path.basename(audio_record.audio_file_path) + ".md")
    shutil.copyfile(audio_record.transcript_file_path, markdown_file)

    

def handle_transcribed(audio_record):
    # Logic for handling transcribed status
    # Summarize transcript, update status, etc.
    with open(audio_record.transcript_file_path, 'r') as f: 
        transcript = f.read()

    model=GPT_MODEL
    summary = summarize_transcript(transcript, model)

    # Save the summary to a file
    summary_file = f"./data/summaries/{os.path.basename(audio_record.audio_file_path)}.txt"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    with open(summary_file, 'w') as f:
        f.write(summary)

    audio_record.transcript_summary_path = summary_file
    audio_record.summary_model_used = model
    audio_record.status = "summarized"


def handle_summarized(audio_record):
    # assume you're getting called to be force resummarized
    handle_transcribed(audio_record)

def handle_error(audio_record):
    # Logic for handling error status
    # Error recovery or notification
    raise SystemError("There was an error for some reason. Time to debug")

def handle_youtube_url(url, should_process_video=False, video_fps=1, force_resummarization=False):
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
    
    # Download audio
    audio_file = download_audio_from_youtube_to_file(url)
    audio_data.audio_file_path = audio_file
    audio_data.status = 'downloaded'
    session.commit()

    # Process video if requested
    base64Frames = None
    if should_process_video:
        video_file_path = download_video_from_youtube(url)
        base64Frames = process_video(video_file_path, fps=video_fps)
        audio_data.base64Frames = base64Frames  # You might need to add this field to your YouTubeAudioData model

    # Transcribe audio
    transcript = transcribe_audio(audio_file)
    transcript_file = f"./data/transcripts/{os.path.basename(audio_file)}.txt"
    os.makedirs(os.path.dirname(transcript_file), exist_ok=True)
    with open(transcript_file, 'w') as f:
        f.write(transcript)
    audio_data.transcript_file_path = transcript_file
    audio_data.status = "transcribed"
    session.commit()

    # Copy transcript file to OBSIDIAN_MARKDOWN_FILE_DESTINATION
    markdown_file = os.path.join(OBSIDIAN_MARKDOWN_FILE_DESTINATION, os.path.basename(audio_data.audio_file_path) + ".md")
    shutil.copyfile(audio_data.transcript_file_path, markdown_file)

    # Summarize transcript (and video frames if available)
    summary = summarize_transcript(transcript, GPT_MODEL, base64Frames)

    # Save the summary to a file
    summary_file = f"./data/summaries/{os.path.basename(audio_data.audio_file_path)}.txt"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as f:
        f.write(summary)

    audio_data.transcript_summary_path = summary_file
    audio_data.summary_model_used = GPT_MODEL
    audio_data.status = "summarized"
    session.commit()

    if force_resummarization and status_before == 'summarized':
        handle_transcribed(audio_data)

    session.refresh(audio_data)
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

### returns a list of all records YouTubeAudioData objects
def fetch_all_records():
    session = Session()
    records = session.query(YouTubeAudioData).all()
    return records


def handle_youtube_option():
    url = st.text_input("Enter a YouTube URL:")
    disable_telegram = st.checkbox("Disable posting to Telegram", value=True)
    force_resummarization = st.checkbox("Force resummarization", value=False)
    should_process_video = st.checkbox("Process video content", value=False)
    video_fps = st.number_input("Video processing FPS", min_value=0.1, max_value=30.0, value=1.0, step=0.1, format="%.1f", disabled=not process_video)


    if st.button("Process"):
        if url:
            summary = handle_youtube_url(url, should_process_video, video_fps, force_resummarization)

            if not disable_telegram:
                post_to_telegram(summary)

            st.markdown(f"<p style='font-size: 24px; color: green;'>Done processing {url}.</p>", unsafe_allow_html=True)
            st.markdown(f"**Summary for the YouTube video at {url}:**")
            st.text_area("Summary", summary, height=250)
        else:
            st.warning("Please enter a URL or upload an audio file.")


def handle_audio_file_option():
    audio_file = st.file_uploader("Upload an audio file:", type=['mp3', 'wav', 'm4a'])

    if audio_file:
        # Save the uploaded audio file to a temporary location
        audio_path = f"./data/audio_files/{audio_file.name}"
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        session = Session()
        audio_data = AudioData(
            title="Placeholder Title",
            description="Placeholder description.",
            audio_file_path=audio_path,
            status='downloaded',
            index_date=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )
        session.add(audio_data)

        status_handlers = {
            'downloaded': handle_downloaded,
            'transcribed': handle_transcribed,
            'summarized': handle_summarized,
            'error': handle_error
        }

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
        assert audio_data.status == 'summarized'

        audio_data.load_summary()

        st.markdown(f"**Summary for the uploaded audio file {audio_file.name}:**")
        st.text_area("Summary", audio_data.summary, height=250)


def handle_previous_transcripts_option():
    # Fetch all transcripts from the database
    youtube_audio_data_records = fetch_all_records()  # This function needs to be implemented
    paths = [y.transcript_summary_path for y in youtube_audio_data_records]
    transcript_id = st.selectbox("Select a transcript to chat with:", paths)

    if transcript_id:
        with open(transcript_id, 'r') as f:
            summary = f.read()
        st.markdown(f"**Summary for the selected transcript:**")
        st.text_area("Summary", summary, height=250)

    if summary:
        st.markdown("You can now chat with the summary above. The model will respond as though it is the speaker.")

        """
        WIP
        # Function to upload the file
        def upload_file(file_path):
            response = OpenAI.File.create(
                file=open(file_path, "rb"),
                purpose='assistants'
            )
            return response.id

        # Upload your transcript
        file_id = upload_file('path_to_your_transcript.txt')

        # Create an assistant
        assistant = OpenAI.Assistant.create(
            model="gpt-4-1106-preview",
            tools=[{"type": "code_interpreter"}],
            file_ids=[file_id]
        )
        """

def main():

    # initialize the local database for tracking audio file, transcripts, summaries, etc.
    # schemas are defined in `models.py`
    init_db()

    st.title("YouTube Audio Transcriber, Summarizer, and Chat")

    input_type = st.radio("Choose input type", ("YouTube URL", "Upload an audio file", "Chat with previous transcripts"))

    # First, figure out what mode we're in and call the handler
    if input_type == "YouTube URL":
        handle_youtube_option()
    elif input_type == "Upload an audio file":
        handle_audio_file_option()
    elif input_type == "Chat with previous transcripts":
        handle_previous_transcripts_option()
    else:
        raise NotImplementedError

    print("done")

if __name__ == '__main__':
    main()