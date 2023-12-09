# Telegram-AudioSummaryBot
Simple streamlit interface to allow you to transcribe audio files or youtube videos using whisper, summarize them using gpt-4, then post the audio file and summary to a telegram channel

# Get started

To run the application, follow these steps:

1. Activate the virtual environment:

    ```bash
    source ./.venv/bin/activate
    ```

2. Start the Streamlit application:

    ```bash
    streamlit run ./app.py
    ```


# Future plans - Support for Whisper.cpp

**IMPORTANT LIMITATIONS**: currently runs only with 16-bit WAV files, so make sure to convert your input before running the tool. For example, you can use ffmpeg like this:

`ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav`

https://github.com/ggerganov/whisper.cpp/tree/master#core-ml-support

## Going to use Mac M1 Neural Engine support to speed things up, and save money

I don't have this working yet because of coremlc installation issues