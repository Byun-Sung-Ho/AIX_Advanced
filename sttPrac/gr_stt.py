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


    def generate_feedback(self, opinion, evaluation_criteria, category):
        print(evaluation_criteria)
        if category == "입론":
            template = """
    ======= 입론 피드백 =======
    [용어 정의가 명확한가] : {}
    
    [주장의 근거가 논리적이고 타당한가]: {}
    
    [논거가 객관적이고 신뢰할 수 있는가]: {}
    ========================
    """

        elif category == "변론":
            template = """
    ======= 변론 피드백 =======
    [최종 변론이 명확하고 간결한가]: {}
    
    [핵심 포인트가 잘 전달되었는가]: {}
    
    [주장이 타당하고 객관적으로 보이는가]: {}
    ========================
    """
        # 평가 기준과 의견을 결합하여 모델에 전달
        prompt = f"평가 기준: {evaluation_criteria}\n의견: {opinion}"
        # 각 의견에 대한 피드백 생성
        response = self.model.generate_content(prompt, safety_settings={'HARASSMENT':'block_none'})
        # 피드백 문자열 반환
        feedback = response.text
        return template.format(feedback, feedback, feedback)

    def FSL(self):
        # few-shot 학습용 데이터
        few_shot_samples = [

            # 입론 샘플
            "input: 인공지능 기술은 사회적 발전과 혁신에 긍정적인 영향을 미치고 있습니다. 개인정보 보호와 노동시장의 변화는 기술 발전과 함께 발생하는 당연한 부작용일 뿐입니다. 이러한 문제들에 대한 대응은 기술 발전의 진보와 함께 이루어질 것이며, 규제보다는 기술적 해결책을 우선적으로 고려해야 합니다. 또한 인공지능 기술의 발전은 노동시장에서 일자리의 변화를 가져올 수 있지만, 새로운 일자리 창출과 함께 사회 전반에 긍정적인 영향을 미칠 것으로 기대됩니다 ",
            "output:======= 입론 피드백 =======\n\n[용어 정의가 명확한가] :\n개인정보 보호와 노동시장 변화에 대한 당연한 부작용으로 언급되었으며, 규제 대신 기술적 해결책을 우선 고려해야 한다는 주장이 명확하게 제시되었습니다.\n\n[주장의 근거가 논리적이고 타당한가]:\n공유된 주장은 인공지능 기술 발전이 노동시장에서의 일자리 변화를 초래할 수 있지만, 새로운 일자리 창출과 함께 사회적 혁신과 발전을 이끌 것으로 예상됩니다.\n\n[논거가 객관적이고 신뢰할 수 있는가]: \n주장은 주로 전반적인 사회적 혁신과 발전에 대한 예상에 기반하고 있으나, 구체적인 예시나 연구 결과를 인용하여 객관성과 신뢰성을 강화할 필요가 있습니다.\n========================"
    
    
            # 반론 샘플
            "input:  인공지능 기술이 사회적 발전과 혁신에 부정적인 영향을 미칠 수 있다는 관점은 다양한 측면에서 제기됩니다. 먼저, 인공지능 기술의 발전은 일부 직업을 자동화하고 일자리를 감소시킬 수 있습니다. 특히, 저숙련 노동자나 노동 집약적인 산업에서는 이러한 자동화로 인한 일자리 감소가 경제적 불안정성을 야기할 수 있습니다. 이는 사회적으로 취약한 계층에 대한 경제적 어려움을 더욱 악화시킬 수 있습니다."
            "output:======= 반론 피드백 =======\n\n[반박의 논리가 명확한가]:\n반박은 명확하게 제기되었으며, 인공지능 기술이 경제적 불안정성을 야기할 수 있다는 관점에 대해 직접적으로 대응하고 있습니다.\n\n[주장에 대한 정보나 근거가 충분한가]:\n반박은 경제적 불안정성과 관련하여 자동화로 인한 일자리 감소가 어떤 영향을 미칠 수 있는지에 대한 충분한 정보를 제공하고 있습니다.\n\n[반박이 원문의 핵심에 집중하고 있는가]:\n반박은 인공지능 기술이 사회적 발전과 혁신에 부정적인 영향을 미칠 수 있다는 원문의 주장에 집중하고 있으며, 특히 경제적 불안정성 측면에 대해 반박하고 있습니다.\n========================"
    
    
    
            # 변론 샘플
            "input:  인공지능 기술은 일부 사람들에게는 사회적 불안과 경제적 어려움을 가져올 수 있다는 우려가 있습니다. 그러나 이러한 우려는 적절한 규제와 기술적 대책을 통해 해결될 수 있습니다. 더불어 인공지능 기술은 혁신과 발전을 이끌어내는 동시에 새로운 경제적 기회와 사회적 이점을 제공할 수 있습니다. 따라서 우리는 이러한 기술의 잠재력을 최대한 발휘하면서도 그에 따른 부정적인 영향을 줄이기 위해 노력해야 합니다."
            "output: ======= 변론 피드백 =======\n\n[최종 변론이 명확하고 간결한가]:\n최종 변론은 인공지능 기술의 부정적 영향과 그에 대한 해결책을 명확하고 간결하게 요약하고 있습니다. 문장 구조도 간결하고 이해하기 쉽습니다.\n\n[핵심 포인트가 잘 전달되었는가]:\n변론에서 주장한 내용과 그 근거가 명확하게 전달되었습니다. 인공지능 기술의 부정적 영향에 대한 우려와 함께 적절한 대응책을 강조하고 있으며, 이를 뒷받침하는 구체적인 예시나 이유가 잘 드러나 있습니다.\n\n[주장이 타당하고 객관적으로 보이는가]:\n변론은 주장과 반박을 근거와 함께 제시하여 타당하고 객관적으로 보입니다. 주장의 강도를 너무 과장하지 않고, 관련된 정보와 근거를 제공하여 신뢰성을 높였습니다.\n========================"
    
    
            # 입론에 대한 미적절한 반박
            "input:  교육비 부담의 정당성과 현실성에 대해 입론에서 제시한 논점을 반박하고자 합니다. 우선, 국가가 교육비를 부담하는 것이 부당하지 않다고 할 수 있습니다. 국가는 교육을 국민에게 제공함으로써 국민의 인권을 보장하고 국가의 경제 발전에 기여합니다. 따라서 국가가 교육비를 부담하는 것은 타당합니다. 또한, 국가가 교육비를 부담함으로써 사회적 안정을 유지할 수 있습니다. 교육은 개인 능력의 실현과 사회의 안정에 밀접한 관련이 있습니다. 따라서 국가가 교육비를 부담함으로써 사회적 안정을 유지하는 것은 합리적입니다.",
            "output: 입론에서의 주장과는 달리, 주제를 논의하기에 미적절한 반박입니다. 주장을 뒷받침하는 대신, 교육비 부담의 정당성과 현실성을 단순히 재확인하는 내용으로 보입니다. 따라서 이는 효과적인 반박이라고 볼 수 없습니다."
        ]
        # few-shot 학습 적용
        self.model.generate_content(few_shot_samples)
    def setting(self, debateRoll, debateTime, debateSide):
        self.category = debateRoll
        self.debateTime = debateTime
        self.debateSide = debateSide
        if self.category == "최종변론":
            self.criteria = "[최종 변론이 명확하고 간결한가]\n[핵심 포인트가 잘 전달되었는가]\n[주장이 타당하고 객관적으로 보이는가]"
        else: self.criteria = "[용어 정의가 명확한가]\n[주장의 근거가 논리적이고 타당한가]\n[논거가 객관적이고 신뢰할 수 있는가]"
        print(self.criteria)
        print(self.category)
        return self.category
    def respond(self, opinion, chat_history):  # 채팅봇의 응답을 처리하는 함수를 정의합니다.
        print()
        bot_message = self.generate_feedback(opinion, self.criteria, self.category)
        chat_history.append((opinion, bot_message))  # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가합니다.
        time.sleep(1)  # 응답 간의 시간 지연을 생성합니다. 이는 봇이 실시간으로 답변하고 있는 것처럼 보이게 합니다.
        return "", chat_history  # 수정된 채팅 기록을 반환합니다.

    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # model = genai.GenerativeModel('gemini-pro')
    # FSL()
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