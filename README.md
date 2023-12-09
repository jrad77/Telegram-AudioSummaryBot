# Telegram-AudioSummaryBot
Simple bot to transcribe audio files, summarize them, then post the audio file and summary to a telegram channel

# Support for Whisper.cpp

**IMPORTANT LIMITATIONS**: currently runs only with 16-bit WAV files, so make sure to convert your input before running the tool. For example, you can use ffmpeg like this:

`ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav`

https://github.com/ggerganov/whisper.cpp/tree/master#core-ml-support

## Going to use Mac M1 Neural Engine support to speed things up, and save money

I don't have this working yet because of coremlc installation issues