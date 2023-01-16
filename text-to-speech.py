#necessary install
#pip3 install --user --upgrade google-cloud-texttospeech

import google.cloud.texttospeech as tts
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="key.json"


def text_to_wav(voice_name: str, text: str):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input, voice=voice_params, audio_config=audio_config
    )

    filename = f"{text}.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')
 


text_to_wav("en-US-Wavenet-A", "Yes")
text_to_wav("en-US-Wavenet-A", "No")
text_to_wav("en-US-Wavenet-A", "Thank you")
text_to_wav("en-US-Wavenet-A", "Goodbye")
text_to_wav("en-US-Wavenet-A", "Hello")
text_to_wav("en-US-Wavenet-A", "Please")
text_to_wav("en-US-Wavenet-A", "I love you")
text_to_wav("en-US-Wavenet-A", "You are welcome")
text_to_wav("en-US-Wavenet-A", "Sorry")