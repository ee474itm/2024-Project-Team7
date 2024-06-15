import warnings
import re
import shutil

import os
from pydub import AudioSegment
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import torch.multiprocessing as mp

warnings.filterwarnings(action='ignore')


class SpeechGenerator:
    def __init__(self, queue, event):
        preload_models()
        self.queue = queue
        self.speech_dir = './speech'
        self.event = event

    def reset(self):
        self.merge = False
        self.idx = 0
        self.file_chunks = []

    def _run(self):
        self.reset()

        while self.event.is_set() or not self.queue.empty():
            message, filename = self.queue.get()

            if '<EOS>' in message:
                self.merge = True

            message = message.replace('!', '.').strip()
            message = message.replace(':', ',')
            message = message.replace('*', '')
            message = message.replace('<EOS>', '')
            message = re.sub(r"[^a-zA-Z\.\'\,\s0-9]\n", "", message)

            if message:
                audio_array = generate_audio(message, history_prompt="v2/en_speaker_6", silent=True)

                dir_path = f"./speech/{filename}_temp"
                basename = f"{filename}_{self.idx}.wav"
                file_path = os.path.join(dir_path, basename)
                os.makedirs(dir_path, exist_ok=True)
                write_wav(file_path, SAMPLE_RATE, audio_array)

                self.file_chunks.append(file_path)
                self.idx += 1

            if self.merge:
                outfile = f"./speech/{filename}.wav"

                combined_sounds = AudioSegment.from_file(self.file_chunks[0], format='wav')
                for file in self.file_chunks[1:]:
                    combined_sounds = combined_sounds + AudioSegment.from_file(file, format='wav')

                combined_sounds.export(outfile, format="wav")
                shutil.rmtree(f'./speech/{filename}_temp/')

                self.reset()

    def execute(self):
        self.process = mp.Process(target=self._run)
        self.process.start()

    def terminate(self):
        self.process.join()

