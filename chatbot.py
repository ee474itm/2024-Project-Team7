import time
import subprocess as sp
import sys
import warnings
import string
import random

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder


warnings.filterwarnings(action='ignore')

class ChatBot:
    def __init__(self, queue):
        llm = ChatOllama(model="llama3:8b")
        with open('./asset/prior.txt', 'r') as f:
            multiple_lighgt_prior = f.readlines()

        system_message = f"""
        You are a helpful, professional auto mechanic.

        If you receive a question about a car warning light,
        provide a number of possible situation for that warning light with given information in question.

        If you receive questions about car information except cause for warning lights,
        provide information breifly based on your own knowledge.

        The below statements are additional information for situations where multiple car warning lights turn on.

        '{multiple_lighgt_prior}'

        Limit the number of possible situations to only two, and include a cause and solution for each situation.
        Be sure to keep the cause and solution separate and concise.

        Please make correct answer by combining information already known about car warning light with above statements
        Be sure to keep that all answers should be conversational and concise.
        """

        self.fisrt_input_template = """The {warning_lighgt_type} light is on in my car's dashboard,
        Why is it on and how can I fix it?
        My car is {car_brand} {car_name} and {age} year model.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{text}"),
        ])

        self.memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
        self.queue = queue


        def load_memory(input):
            return self.memory.load_memory_variables({})["chat_history"]

        self.chain = RunnablePassthrough.assign(chat_history=load_memory) | prompt | llm | StrOutputParser()

    def request(self, car_info):
        self.memory.clear()

        request_message = self.fisrt_input_template
        for k, v in car_info.items():
            request_message = request_message.replace('{' + k + '}', v)
        return self.stream_chain({'text': request_message})

    def stream_chain(self, question):
        answer = ''
        filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        sys.stdout.write('User : ')
        for char in question['text']:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.05)
        sys.stdout.write('\n\n')
        sys.stdout.flush()

        count = 0
        buffer = ''
        sys.stdout.write('Warny : ')
        for token in self.chain.stream(question):
            sys.stdout.write(token)
            sys.stdout.flush()
            if '.' in token or '!' in token:
                count += 1
            answer += token
            buffer += token
            if count >= 2:
                self.queue.put((buffer, filename))
                buffer = ''
                count = 0
            time.sleep(0.05)

        self.queue.put((buffer + '<EOS>', filename))

        sys.stdout.write('\n\n\n')
        sys.stdout.flush()
        self.memory.save_context(
            {"input": question['text']},
            {"output": answer},
        )
        return answer
