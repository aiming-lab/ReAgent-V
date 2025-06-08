
def get_asr_docs(video_path, audio_path):

    full_transcription = []
    try:
        extract_audio(video_path, audio_path)
    except:
        return full_transcription
    audio_chunks = chunk_audio(audio_path, chunk_length_s=30)
    
    for chunk in audio_chunks:
        transcription = transcribe_chunk(chunk)
        full_transcription.append(transcription)

    return full_transcription

