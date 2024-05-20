import gradio as gr  # 그라디오 라이브러리를 불러옵니다.
import random  # 무작위 선택을 위한 라이브러리를 불러옵니다.
import time  # 시간 지연을 위한 라이브러리를 불러옵니다.

import speech_recognition as sr
import os
import time
import pyaudio
import wave
import google.generativeai as genai
import openai
import os

class gr_interface:
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        self.demo = gr.Blocks()
        self.criteria = ""
        self.category = ""
        self.debateTime=10
        self.debateSide=""
        self.speakingText = ""
        self.attack = """
[공격적인 어조 확인]
- 주어진 글에 상대방을 비방하거나 공격적인 어조가 있는가.

[공격적인 어휘 사용 확인]
- 주어진 글에 특정 어휘나 표현이 상대방에게 공격적으로 향하고 있는가.

[공격성 점수 확인]
- 주어진 글에서 토론 태도에 따른 공격성이 0부터 10까지의 범위 중 어느 정도인지 점수를 매겨주세요.
"""

    def generate_feedback(self, opinion, evaluation_criteria, category):
        if category == "입론":
            template = """
    ======= 입론 피드백 =======
    {}
    ========================
    """

        elif category == "변론":
            template = """
    ======= 변론 피드백 =======
    {}
    ========================
    """

        elif category == "공격성 평가":
            template = """
    ======= 공격성 피드백 =======
    {}
    ========================
    """

        # 평가 기준과 의견을 결합하여 모델에 전달
        prompt = f"평가 기준: {evaluation_criteria}\n의견: {opinion}"
        # 각 의견에 대한 피드백 생성
        response = self.model.generate_content(prompt, safety_settings={'HARASSMENT': 'block_none'})
        # 피드백 문자열 반환
        feedback = response.text
        return template.format(feedback, feedback, feedback)

    def detect_aggression_with_count(self, input_text):
        aggressive_words = ["시발", "씨발", "시바", "개새끼", "미친놈", "미친새끼", "병신", "ㅅㅂ", "ㅆㅂ"]
        aggression_count = 0
        aggressive_word_list = []
        for word in aggressive_words:
            if word in input_text:
                aggression_count += input_text.count(word)
                aggressive_word_list.append(word)
        self.model.generate_content("너는 공격성 평가를하는 심사위원이다. 앞으로 받을 글에서 나온 내용을 기반으로 평가를 진행하라. 평가를 할때는 반드시 글에 있는 내용을 참조하라", safety_settings={'HARASSMENT': 'block_none'})
        prompt = f"평가 기준: {self.attack}\n주어진 글: {input_text}"
        response = self.model.generate_content(prompt, safety_settings={'HARASSMENT': 'block_none'})
        feedback = response.text
        template = """
    ======= 공격성 피드백 =======
    {}
    ========================
    """
        return template.format(feedback, feedback, feedback)

        # if aggression_count > 0:
        #     print("[공격적인 단어가 있는가]: 네, 공격적인 단어가 있습니다.")
        #     print("[어떤 단어가 공격적인가]:", ', '.join(aggressive_word_list), "이(가) 공격적으로 사용되었습니다.")
        #     print("[공격성 점수]:", aggression_count)
        #     return self.model.generate_content(self.attack, safety_settings={'HARASSMENT': 'block_none'})
        # else:
        #     print("[공격적인 단어가 있는가]: 아니요, 공격적인 단어가 없습니다.")
        #     return "[공격적인 단어가 있는가]: 아니요, 공격적인 단어가 없습니다."

    def FSL(self):
        # few-shot 학습용 데이터
        # few-shot 학습용 데이터
        few_shot_samples = [

            # 찬성측 입론 샘플 (2016 인천고등학생 토론대회)
            """input:
            찬성측 입론 시작하겠습니다.얼마전 대전에서 택시기사분의 급작스러운 심장마비 증세로 인한 교통사고가 있었습니다. 하지만 탑승하고 있던 승객들은 어떠한 구조조치 없이 신고는 처녕 귀중품만 훔친체 그들의 골프여행을 떠났다고 합니다.
            택시기사를 죽음으로 몰고간 그들은 어떻게 보면 가해자가 아닐까 라는 생각이 듭니다. 이후 국내에서도 착한 사마리안 법을 제정하자는 목소리가 커지고 있으며 이에 찬성하는바이며 이유는 다음과 같습니다.
            이기적인 사회인 현대 사회의 그늘진 실상을 극복하자는 의미인 착한 사마리안 법은 비인간화 비윤리화된 사회와 법에 대한 새로운 윤리학을 의미하는 것입니다.
            둘째 많은 나라에서 착한 사마리안 법을 제정함으로써 좋은 효과를 보았으며 대 내외적으로 효과가 증명되었습니다. 셋째 법으로 최소한의 사회적 윤리를 보호해주어야 합니다. 인간성을 위배한 행위를 단지 윤리적인 문제로면 남겨두어서는 안됩니다.
            도덕적 의무도 법이 보호해야할 중요한 가치 중 하나로 유지해야한다고 생각합니다.
            """,

            """output:
            [용어 정의가 명확한가] : 찬성측은 '착한 사마리안 법'을 명확하게 정의하고 있으며, 택시 사고 사례를 통해 그 필요성을 제시하고 있습니다.
            [주장의 근거가 논리적이고 타당한가]: 주장은 택시 사고 사례를 통해 착한 사마리안 법의 필요성을 논리적으로 제시하고 있습니다. 또한 다른 나라의 사례를 인용하여 법의 효과를 설명하고 있으며, 법이 보호해야 할 사회적 윤리와 인간성에 대한 중요성을 강조하고 있습니다.
            [논거가 객관적이고 신뢰할 수 있는가]: 주장은 실제 사례와 다른 나라의 경험을 통해 착한 사마리안 법의 효과를 논리적으로 뒷받침하고 있습니다. 또한 법이 보호해야 할 사회적 윤리에 대한 이해와 인간성을 강조하여, 주장이 객관적이고 신뢰할 수 있는 논거를 제시하고 있습니다.
            """,

            # 뱐대측 입론 샘플 (2016 인천고등학생 토론대회)
            """input:
            반대측 입론 시작하겠습니다. 최근 우리나라의 사회가 각박해지고 있는것은 사실입니다. 하지만 저희는 법이 해결측이 되어줄 수 없고 되어서도 안된다고 생각함으로 구조를 의무화해야한다는 측에 대해 반대합니다.
            첫번째로 법은 현사회가 즉면한 문제점의 근본적인 해결측이 되어주지 못합니다. 두번째는 구조 의무화의 역설입니다. 구조 의무화는 결과적으로 국민들이 사고를 외면하는 역설적 상황을 만들어 낼 수 있습니다.
            저희 팀도 적극적 구조가 이루어져야한다는 것에 대해서는 동의합니다. 하지만 법이라는 타율적인 구조의 의무화 이외에도 다른 근본적인 방안이 존재하고 법의 효력도 위배할 가능성이 큼으로 본 주제에 대해서 반대하는 바입니다.
            """,

            """output:
            [용어 정의가 명확한가] : 반대측은 '구조 의무화'에 대한 의미를 명확하게 정의하고 있으며, 법의 효력과 역설에 대한 관점을 제시하고 있습니다.
            [주장의 근거가 논리적이고 타당한가]: 주장은 법이 현사회의 근본적인 문제를 해결할 수 없음을 논리적으로 제시하고 있습니다. 또한 구조 의무화가 역설적인 상황을 만들 수 있다는 관점을 통해 법의 효력을 의심하고 있습니다.
            [논거가 객관적이고 신뢰할 수 있는가]: 주장은 법의 한계와 구조 의무화의 역설에 대한 관점을 제시하면서, 객관적이고 신뢰할 수 있는 논거를 제시하고 있습니다. 또한 다양한 방안을 고려해야 한다는 입장을 통해 다양성과 유연성을 강조하고 있습니다.
            """

            ### 2018 대한민국 열린 토론대회 대학생부 결승전 (의무투표제 실시)
            """input:
            반대측 입론 시작하겠습니다. 맞지 않는 옷 의무투표제를 적절히 나타내는 표현입니다.첫째로 의무 투표 제가 우리나라 실정에 부합하지 않기 때문입니다. 당신이 틀렸습니다. 2016년 중앙선거관리위원회에 따르면 현재 감성 의무 투표제를 도입하고 있는 국가는 8개국 인데요 100만 명 정도로 인구수가 지극히 적은 곳이 3곳 군사독재 후에 무투표제 를 도입한 국가가 4곳입니다. 4곳이라고요
            시발. 인구 수가 적지 않고 현재 민주주의의 성숙도가 높은 우리나라는 투표를 의무로 강제할 필요가 없습니다 나머지 한 곳인 호주는 식민지배 직후 바로 의무투표 제 를 도입한 곳으로 도입 배경에 달라 우리나라에 그대로 적용할 수 없으며  조선일보는 호주의 높은 득표율이 정체 교육과 투표시간 연장 이라는 보안적 장치 를 도입한 결과 라고 밝힌 바 있습니다.둘째로 강제 성으로 인한 투표는 제대로 된 민의 반영을 막기 때문입니다.
            높은 투표율이 반드시 민의에 정확한 반영을 되니 하지 않습니다. 스스로 후보자를 알아보고 공약을 평가하는 것이 아닌 벌금을 회피하기 위한 행위는 큰 의미가 없기 때문입니다 실제로 런던대학교 싸라 볼지 교수는 투표를 의무로 강제할 경우 후보자가 누군지도 모르는 사람들이 선거에 참여하여 투표에 7이 낮아질 수도 있다는 연구결과를 발표한 바 있습니다 따라서 저희 팀은 의무투표제를 반대합니다.
            """,

            """output:
            [용어 정의가 명확한가] : 반대측은 '의무투표제'에 대한 개념을 명확히 이해하고, 현재 우리나라의 상황과 국제적 비교를 통해 그 적합성을 논의하고 있습니다. 의무투표제는 모든 성인 시민에게 투표를 강제하는 제도로, 이에 대한 개념을 정의함과 동시에, 국제적으로 해당 제도가 도입된 사례를 제시하여 의미를 부여하고 있습니다.
            [주장의 근거가 논리적이고 타당한가]: 주장은 의무투표제가 우리나라의 실정과 민의 반영에 어떻게 맞지 않는지를 논리적으로 제시하고 있습니다. 투표의 강제성이 민의 반영을 방해할 수 있다는 점을 논리적으로 뒷받침하고 있으며, 이를 통해 정당한 의견을 제시하고 있습니다. 또한, 런던대학교 교수의 연구 결과를 인용하여 의무투표제의 부작용에 대한 가능성을 제기하고 있으며, 이를 통해 주장이 타당하다는 점을 강조하고 있습니다.
            [논거가 객관적이고 신뢰할 수 있는가]: 주장은 중앙선거관리위원회의 데이터를 인용하여 현재의 국제적 추세를 제시하고 있으며, 이를 통해 의무투표제의 전반적인 상황을 객관적으로 보여주고 있습니다. 런던대학교 교수의 연구 결과를 인용하여 의무투표제의 부작용을 신뢰할 수 있는 논거로 제시하고 있으며, 이러한 객관적인 자료와 연구 결과를 바탕으로 주장이 신뢰할 수 있다는 점을 강조하고 있습니다. 이에 따라 주장은 객관적이고 타당하며 신뢰할 수 있는 주장임을 입증하고 있습니다.
            """
        ]
        # few-shot 학습 적용
        self.model.generate_content(few_shot_samples)

    def setting(self, debateRoll, debateTime, debateSide):
        self.category = debateRoll
        self.debateTime = debateTime
        self.debateSide = debateSide
        if self.category == "최종변론":
            self.criteria = """
[최종 변론이 명확하고 간결한가]
- 최종 변론이 명확하고 간결하게 전달되었는지 확인해주세요.

[핵심 포인트가 잘 전달되었는가]
- 핵심 포인트가 잘 전달되었는지 확인해주세요.

[주장이 타당하고 객관적으로 보이는가]
- 주장이 타당하고 객관적으로 보이는지 확인해주세요.
"""
        else: self.criteria = """
[용어 정의가 명확한가]
- 토론에서 사용된 용어의 정의가 명확하게 전달되었는지 확인해주세요.

[주장의 근거가 논리적이고 타당한가]
- 주장의 근거가 논리적이고 타당한지 확인해주세요.

[논거가 객관적이고 신뢰할 수 있는가]
- 논거가 객관적이고 신뢰할 수 있는지 확인해주세요.
"""
        print(self.criteria)
        print(self.category)
        return self.category

    def respond(self, opinion, chat_history):  # 채팅봇의 응답을 처리하는 함수를 정의합니다.
        bot_message = self.generate_feedback(opinion, self.criteria, self.category)
        chat_history.append((opinion, bot_message))  # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가합니다.
        time.sleep(1)  # 응답 간의 시간 지연을 생성합니다. 이는 봇이 실시간으로 답변하고 있는 것처럼 보이게 합니다.
        attack_bot_message = self.detect_aggression_with_count(opinion)
        chat_history.append(("공격성 검사", attack_bot_message))  # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가합니다.
        time.sleep(1)  # 응답 간의 시간 지연을 생성합니다. 이는 봇이 실시간으로 답변하고 있는 것처럼 보이게 합니다.
        return "", chat_history  # 수정된 채팅 기록을 반환합니다.

    def launch_interface(self):
        self.setup_interface()
        self.demo.launch(share=True)

    def ver2(self):
        # 오디오 녹음 설정
        FORMAT = pyaudio.paInt16  # 16비트 PCM 형식으로 녹음
        CHANNELS = 1  # 단일 채널(모노)로 녹음
        RATE = 44100  # 샘플링 속도(Hz)
        CHUNK = 1024  # 버퍼 크기

        # 오디오 녹음 시간 설정 (초)
        RECORD_SECONDS = self.debateTime
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
                    f.write(str(text) + '\n')
                self.speakingText = text
                print(self.speakingText)
                return self.speakingText


                # print(text)
            except Exception as e:
                print("Exception: " + str(e))

    def setup_interface(self):
        with self.demo:
            self.FSL()
            chatbot = gr.Chatbot(label="토론창")  # '채팅창'이라는 레이블을 가진 채팅봇 컴포넌트를 생성합니다.
            msg = gr.Textbox(label="토론 내용")  # '입력'이라는 레이블을 가진 텍스트박스를 생성합니다.
            with gr.Row():
                # debateRoll = gr.Textbox(label="입론 or 최종변론 or 반론 입력")
                # debateTime = gr.Textbox(label="말하는 시간 입력(초)")
                # debateSide = gr.Textbox(label="찬/반 입력")
                debateRoll = gr.Dropdown(["입론", "최종변론", "반론"],label="역할 선택")
                debateTime = gr.Dropdown([10, 2, 3, 4, 5], label="말하는 시간 선택(분)")
                debateSide = gr.Dropdown(["찬성", "반대"],label="찬반 선택")
                settingButton = gr.Button("완료")
            with gr.Row():
                start = gr.Button("녹음 시작")
                clear = gr.Button("초기화")  # '초기화'라는 레이블을 가진 버튼을 생성합니다.

            start.click(self.ver2, [],[msg])
            msg.submit(self.respond, [msg, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출되도록 합니다.
            settingButton.click(self.setting, [debateRoll,debateTime,debateSide])
            clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼을 클릭하면 채팅 기록을 초기화합니다.

if __name__ == "__main__":
    poc_interface = gr_interface()
    poc_interface.launch_interface()