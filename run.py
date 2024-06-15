from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
import time
import subprocess as sp
import sys
import torch.multiprocessing as mp
import warnings
import torch

from tts import SpeechGenerator
from detection import Detector
from chatbot import ChatBot

warnings.filterwarnings(action='ignore')


def save_text(text, filename):
    with open(f'{filename}.txt', 'w') as writer:
        writer.write(text)


def run_sample():
    car_info = {
        "image_path": 'images/sample.jpg',
        # "warning_lighgt_type": "Airbag Warning and Stability Control Off",
        "car_brand": "Fort",
        "car_name": "Fusion",
        "age": "2015",
    }

    sp.run(['catimg', car_info['image_path']])

    results = detector.run('images/sample.jpg')
    car_info['warning_lighgt_type'] = " and ".join(results[0])
    print(car_info)
    request = chatbot.request(car_info)
    answer = chatbot.stream_chain({
        'text': 'Are there other possible causes and solutions for these warning lights?'
    })
    answer = chatbot.stream_chain({
        'text': 'How can excessive voltage affect the ECU and airbag module with wiring issues?'
    })


if __name__ == "__main__":
    mp.set_start_method("spawn")

    event = mp.Event()
    queue = mp.Queue()
    detector = Detector()
    tts_model = SpeechGenerator(queue, event)
    chatbot = ChatBot(queue)

    event.set()
    tts_model.execute()

    run_sample()

    event.clear()
    tts_model.terminate()
