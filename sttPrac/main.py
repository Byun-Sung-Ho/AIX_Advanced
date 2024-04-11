import speech_recognition as sr
import os
import time
import pyaudio
import wave
import google.generativeai as genai
import openai
import os

def getAudio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = " "

        try:
            said = r.recognize_google(audio, language='ko-KR')
            print(said)
        except Exception as e:
            pass
            # print("Exception: " + str(e))
    return said

def ver1():
    while True:
        text = getAudio()
        with open('memo.txt', 'a') as f:
            f.write(str(text)+'\n')

        if "이상입니다" in text:
            break

        time.sleep(0.05)
def ver2():
    # 오디오 녹음 설정
    FORMAT = pyaudio.paInt16  # 16비트 PCM 형식으로 녹음
    CHANNELS = 1  # 단일 채널(모노)로 녹음
    RATE = 44100  # 샘플링 속도(Hz)
    CHUNK = 1024  # 버퍼 크기

    # 오디오 녹음 시간 설정 (초)
    RECORD_SECONDS = 10
    # 저장할 오디오 파일 이름
    WAVE_OUTPUT_FILENAME = "output.wav"

    # PyAudio 객체 생성
    audio = pyaudio.PyAudio()

    # 마이크 입력 설정
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("녹음 시작...")

    frames = []

    # 마이크로부터 오디오 데이터 읽기
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("녹음 종료.")

    # 스트림 닫기
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 오디오 데이터를 WAV 파일로 저장
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print("오디오 파일 저장 완료:", WAVE_OUTPUT_FILENAME)

    audio_file_path = "output.wav"
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio_data, language="ko-KR")
            with open("memo.txt", 'w') as f:
                f.write(str(text)+'\n')
            return text
            # print(text)
        except Exception as e:
            print("Exception: " + str(e))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content(ver2())
print(response.text)