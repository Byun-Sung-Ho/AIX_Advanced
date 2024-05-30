import os
from google.cloud import speech, storage
from google.protobuf.json_format import MessageToDict
from pydub import AudioSegment
from kiwipiepy import Kiwi
import pandas as pd

# Google Cloud 서비스 계정 키 파일 경로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ".\gen-lang-client-0969588048-889a2e0ba98c.json"

def convert_to_mono_and_resample(source_path, destination_path):
    """오디오 파일을 모노로 변환하고 샘플 속도를 48000 Hz로 변환"""
    audio = AudioSegment.from_wav(source_path)
    audio = audio.set_channels(1)  # 모노로 변환
    audio = audio.set_frame_rate(48000)  # 샘플 속도 변환
    audio.export(destination_path, format="wav")
    print(f"Converted {source_path} to mono and resampled to 48000 Hz as {destination_path}")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """파일을 Google Cloud Storage 버킷에 업로드"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def transcribe_gcs(gcs_uri):
    """Google Cloud Storage에 저장된 오디오 파일을 텍스트로 변환"""
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="ko-KR",
        enable_word_time_offsets=True
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    print("Waiting for operation to complete...")
    response = operation.result(timeout=10000)

    return response

def process_audio_file(audio_file_path, bucket_name, gcs_filename, output_csv_path="timestamps.csv", output_audio_folder="segmented_audio"):
    # 모노 및 샘플 속도 변환
    mono_audio_file_path = "mono_" + audio_file_path
    convert_to_mono_and_resample(audio_file_path, mono_audio_file_path)

    # Google Cloud Storage에 오디오 파일 업로드
    upload_to_gcs(bucket_name, mono_audio_file_path, gcs_filename)
    gcs_uri = f"gs://{bucket_name}/{gcs_filename}"

    # GCS URI를 사용하여 음성 인식 수행
    response = transcribe_gcs(gcs_uri)
    print("음성을 텍스트로 변환 완료")

    # 변환된 텍스트 추출 및 타임스탬프 정보 가져오기
    words_info = []
    for result in response.results:
        alternative = result.alternatives[0]
        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time.total_seconds()
            end_time = word_info.end_time.total_seconds()
            words_info.append((word, start_time, end_time))

    full_text = " ".join([word[0] for word in words_info])

    # Kiwi를 사용하여 문장 분리
    kiwi = Kiwi()
    sentences = kiwi.split_into_sents(full_text)

    # 출력 오디오 폴더가 없으면 생성
    if not os.path.exists(output_audio_folder):
        os.makedirs(output_audio_folder)

    output_data = []
    audio = AudioSegment.from_wav(mono_audio_file_path)
    current_index = 0

    for i, sentence in enumerate(sentences):
        sentence_text = sentence.text.strip()
        sentence_words = sentence_text.split()

        # 문장의 시작 시간과 끝 시간 계산
        start_time_sentence = None
        end_time_sentence = None

        for word_info in words_info[current_index:]:
            if word_info[0] == sentence_words[0] and start_time_sentence is None:
                start_time_sentence = word_info[1]
            if word_info[0] == sentence_words[-1]:
                end_time_sentence = word_info[2]
                break

        # 오디오 파일 분할
        start_ms = start_time_sentence * 1000  # 밀리초로 변환
        end_ms = end_time_sentence * 1000  # 밀리초로 변환
        segment = audio[start_ms:end_ms]

        # 분할된 오디오 파일 저장
        segment_filename = f"{output_audio_folder}/sentence_{i + 1}.wav"
        segment.export(segment_filename, format="wav")

        output_data.append([i + 1, segment_filename, sentence_text])
        print(f"문장 {i+1}: {sentence_text}, 파일 저장: {segment_filename}")

        current_index += len(sentence_words)

    # CSV 파일로 저장
    df = pd.DataFrame(output_data, columns=["ID", "PATH", "TEXT"])
    df.to_csv(output_csv_path, index=False, encoding='utf-8')

# 오디오 파일 경로와 출력 CSV 경로 및 출력 오디오 폴더 지정
audio_file_path = '[2018 대한민국 열린 토론대회 입론].wav'  # 오디오 파일 경로
bucket_name = 'onto_bucket'  # Google Cloud Storage 버킷 이름
gcs_filename = 'audio_file.wav'
output_csv_path = 'timestamps.csv'
output_audio_folder = 'segmented_audio'

# 전체 프로세스 실행
process_audio_file(audio_file_path, bucket_name, gcs_filename, output_csv_path, output_audio_folder)


