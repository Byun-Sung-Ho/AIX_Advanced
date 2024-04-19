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

def generate_feedback(opinion, evaluation_criteria):
    feedback = []
    # 평가 기준과 의견을 결합하여 모델에 전달
    prompt = f"평가 기준: {evaluation_criteria}\n의견: {opinion}"
    # 각 의견에 대한 피드백 생성
    response = model.generate_content(prompt, safety_settings={'HARASSMENT':'block_none'})
    for part in response.parts:
        # 각 부분의 텍스트를 가져와서 피드백 리스트에 추가
        feedback.append(part.text)
    return '\n'.join(feedback)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')
# few-shot 학습용 데이터
few_shot_samples = [
    "input : 안녕하세요? 오늘의 논제인 ‘교육비 국가가 부담해야 한다’의 긍정측을 맡은 oooo팀 첫번째 토론자 ooo입니다. 토론을 시작하기 전에 저희 긍정측에서는, 오늘의 논의가 보다 발전적인 방향으로 나아갈 수 있도록, 논제의 주요 단어인 ‘교육비’에 대해 정의해보고자 합니다. 교육비에는 공교육비와 사교육비가 있습니다. 이 중 사교육에 드는 비용은, 개인이 자신의 선택에 따라 대가를 지불하는 것이기에 분명히 개인의 몫입니다. 그러므로 저희 긍정 측에서는 오늘의 논제에 ‘교육비’를 학교교육으로 대표되는 공교육에 드는 비용으로 정의하고자 하고, 그것을 제안합니다. 18세기 계몽철학자인 콩도르세는, 인간은 모두 동등한 권리를 가지고 있다는 프랑스 혁명의 기본이념에 기초하여, 공교육의 이념인 평등을 주장하였습니다. 이 주장은 ‘공교육조직 법안’으로 구체화 되었고, 이후 근대에서 현대에 이르는 교육제도를 확립하는 밑바탕이 되었습니다. 이러한 공교육의 기본 이념을, 교육제도를 통해 실현하는 핵심적 요소는, 바로 국가가 교육에 필요한 비용을 부담하는데 있을 것입니다. 저희 긍정측에서는 국가가 교육비를 부담해야 하는 이유에 대해 ‘당위적인 측면’과 ‘현실적인 측면’으로 나누어 말씀드리겠습니다. 먼저 당위적인 측면에서 국가가 교육비를 부담해야 하는 첫번째 이유는, 국민에게 교육받을 권리가 있기 때문입니다. 우리의 헌법31조의 1항은 ‘모든 국민은 능력에 따라 균등하게 교육을 받을 권리를 가진다’라고 명시하고 있습니다. 여기서 말하는 교육을 받을 권리라는 것은 학습권으로써, 교육을 받을 수 있도록 국가에 적극적인 배려를 요구할 수 있는 권리입니다. 그러므로 이 권리는 국가가 교육에 필요한 재정과 시설, 제도 등을 실질적으로 갖추어줌으로서 보장될 수 있습니다. 두번째로 교육비를 부담하여 평등한 교육을 보장하는 것은 국가의 의무이기도 합니다. 우리의 헌법은 교육에 대해 국민의 권리의 부분뿐 아니라 국가의 의무에 대해서도 규정하고 있습니다. 헌법 31조 6항에 언급된 것처럼 교육제도운영과 교육재정에 관해서는 법률로 정하여 실효성을 갖도록 하고 있습니다. 즉 다시 말해서 국가는, 사회 경제적 약자도 능력에 따라 실질적 평등교육을 받을 수 있도록, 교육비 부담과 같은 적극적인 정책을 실현해야할 의무가 있다는 것입니다. 그러나 현재 우리 국가가 부담하는 교육비 비율은, 다른 나라들과 비교해 볼 때 상당히 낮은 편입니다. OECD 교육지표에 따르면 2002년 한국의 국가부담 공교육비는 GDP 대비 4.2퍼센트로, 이는 OECD국가들의 평균인 5.1퍼센트에 비해 가장 낮은 수준에 속합니다.(또한 이는 4.6퍼센트를 부담하고 있는 비OECD 국가인 태국보다도 교육비의 국가 부담율이 낮다는 것을 의미합니다.) 다음으로 현실적인 측면에서 교육비를 국가가 부담해야 하는 첫번째 이유는, 교육의 양극화로 인한 사회계층의 고착화를 막기 위해서 입니다."
    "output : **용어 정의**\n    - 명확하고 정확함: 교육비를 학교 교육에 드는 공교육 비용으로 정의함.\n**주장의 근거**\n    1. 논리적이고 타당함:\n        - 공교육의 평등 원칙 (콩도르세)\n        - 헌법에 근거한 교육 받을 권리와 국가의 교육 의무\n        - 한국의 낮은 국가 교육비 부담률\n    2. 타당성에 대한 의문:\n        - '사회계층 고착화'와 '사회 안정'이 교육비 국가 부담의 직접적인 원인인지에 대한 논의가 부족함.\n**논거의 객관성 및 신뢰성**\n    1. 객관적이지 않음:\n        - 현재 교육 불평등의 원인을 교육비 부담률에만 돌림.\n    2. 신뢰할 수 있음:\n        - OECD 교육지표 및 국정감사 보고서와 같은 출처 인용.\n**전반적인 평가**\n    - 용어 정의: 명확함\n    - 주장의 근거: 논리적이고 타당하지만 타당성에 대한 의문이 있음\n    - 논거의 객관성 및 신뢰성: 객관적이진 않지만 신뢰할 수 있음\n\n**개선 방향**\n    - '사회계층 고착화'와 '사회 안정'과의 관계에 대한 논의 추가\n    - 교육의 질에 대한 논의 추가\n    - 교육비 국가 부담의 장점과 단점을 균형 있게 제시\n\n**추가 정보**\n    - 교육부 홈페이지: [https://moe.go.kr/](https://moe.go.kr/)\n    - OECD 교육지표: [https://data.oecd.org/education.htm](https://data.oecd.org/education.htm)\n    - 국정감사 보고서: [https://www.nis.go.kr/](https://www.nis.go.kr/)"
]

# few-shot 학습 적용
response = model.generate_content(few_shot_samples)
# 토론 평가 기준 정의
criteria_1 = "1. 용어 정의가 명확한가?\n2. 주장의 근거가 논리적이고 타당한가?\n3. 논거가 객관적이고 신뢰할 수 있는가?"

# 사용자로부터 찬성 및 반대측 의견 입력 받기
# pro_opinion = ver2()
pro_opinion = ("안녕하세요? 오늘의 주제인 ‘교육비, 국가가 부담해야 한다’의 부정측을 맡은 oooo팀 첫번재 토론자 ooo입니다."

+"철학자 헤겔은, 인류역사를 자유의식의 발전사라 했습니다. 그러나 저희는 인류역사를 교육의 발전사로 말하고 싶습니다. 왜냐하면 교육은 자유의식을 발전시킬 뿐 아니라 사회현실도 반영하기 때문입니다. 교육의 형태는 시대의 요구와 사회적인 필요성에 의해 변화되어 왔습니다. 사회가 농업사회에서 산업사회로, 산업사회에서 다시 현대의 지식 정보사회로 변해감에 따라, 교육체계는 달라져 왔고, 현대 사회에서의 교육은, 그 필요한 인재를 길러 내기 위해 다양성과 자율성을 요구받고 있습니다."

+"저희 oooo팀에서는 교육비를 국가가 전적으로 부담해서는 안된다고 생각합니다. 그 이유에 대해 저희는 크게 세 가지 근거를 들어 말씀드리겠습니다."

+"첫번째로 교육의 다양한 선택권 보장을 위해, 교육비를 국가가 전적으로 부담해서는 안됩니다."

+"현대에 필요한 교육은 과거처럼 국가주도의 획일화된 교육이 아니라, 다양한 선택권을 제공하는 교육입니다. 새로운 지식이 끊임없이 확대 재생산되고 여러 가치관이 공존하는 현대에서, 다양한 교육에 대한 요구는 점점 높아졌고, 이것은 대안 교육이나 특성화 교육에 대한 관심으로 나타나고 있습니다. 대표적 대안학교인 간디학교의 교과과정을 보면 세 개 이상 자립교과과목을 선택할 수 있게 하고 있고, 마은 개 이상의 특기적성 교과과정을 개설하였습니다. 또한 자립형 사립고인 해운대고는 국민공통기본과정에 더하여 보통교과의 일부를 전문 교과로 편성하여, 재량활동을 강화하고 있습니다. 이러한 대안교육과 특성화교육을 하기 위해서는, 국가가 전적으로 비용을 부담하는 체제가 아니라, 개인들의 교육비 부담을 통한 자율성 확립이 필요합니다. 그리고 현재도 그런 형태로 이루어지고 있습니다."

+"두번째로 교육비를 국가가 부담하는 것은, 교육에 있어 부작용을 가져올 수도 있습니다."

+"1970년대부터 대학 완전 무상교육을 해왔던 독일은, 다시 수업료를 도입하고 있습니다. 무상교육이 이루어진 후 독일에서는 대학생이 급증하여, 교원 1인당 학생 수가 과거 약 10명 이하에서 현재 약 35명까지 늘었습니다. 이는 곧 독일학생들이 과거와 같이, 교수님들과 개별적인 학문적 접촉을 할 수 있는 기회가 줄어들었음을 의미합니다. 뿐만 아니라 교육에 필요한 제반시설도 과거에 비해 상당히 부족해졌습니다. 이처럼 무상 교육은 각종 부작용을 나아 독일 대학의 교육여건을 악화시키는 요인으로 작용하고 있습니다."

+"마지막으로 한국의 국가 재정으로 볼 때, 국가가 현재보다 더 많은 교육비를 부담하는 것은, 현실적으로 굉장히 어려운 일입니다."

+"2004년 우리나라의 GDP대비 국가재정규모는 27.3%로, OECD 평균인 40.8%에 크게 못 미치고  OECD국가들 중, 작은 정부를 추구하는 미국의 재정규모 36%와 비교하여도, 훨씬 낮은 수준입니다. 이렇게 다른 국가에 비해 분명히 적은 재정규모에서도 2005년 우리 정부는 총 예산인 131조 5천억원의 약 5분의 1에 해당하는 25조 9400억원을 교육부 예산으로 책정하고 있습니다. 이것은 11조 1800억원을 투자하는 산업분야 지원보다 두 배나 많고, 27조 5천억우너을 투자하는 도로 항만 등의 사회 간접자본 투자와 비슷한 수치입니다."

+"결국 현재 우리나라의 국가 재정으로는, 교육비 부담을 더 이상 늘릴 수 없고, 교육비 부담만을 목표로 무리하게 세금을 올렸다가는 조세저항에 부딪치게 될 것입니다. 즉 교육이 개인에게는 미래를 위한 투자의 일부라고 하지만, 국가의 차원에서는 그것만을 위해 현재의 경제구조나 사회기반을 포기할 수 없기 때문입니다."

+"지금까지 저는 국가가 교육비를 부담해서는 안 되는 이유에 대해서 첫째 학교교육의 다양화, 두번째 교육에서 나타날 수 있는 부작용, 마지막으로 국가교육비 부담의 비현실성이라는 세 가지 근거를 들어 말씀드렸습니다."

+"감사합니다.")
pro_feedback = generate_feedback(pro_opinion, criteria_1)
# response = model.generate_content(ver2())
# print(response.text)
print(pro_feedback)