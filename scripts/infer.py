import time
import logging
from faster_whisper import WhisperModel, decode_audio

class infer():

    def __init__(self, model_name, load={}):
        self.model_name = model_name
        self.load = load
        logging.info(f'LOADING {model_name} ...')
        self.model = WhisperModel(model_name, **load)
        logging.info(f'DONE')

    def __call__(self, audio, fdo, split_stereo=False, transcribe={}):
        self.fdo = fdo
        self.out = []
        self.add(f"model={self.model_name}")
        self.add(f"audio={audio}")
        self.add(f"split_stereo={split_stereo}")
        self.add(f"load={self.load}")
        self.add(f"transcribe={transcribe}")

        logging.info(f'READING {audio} ...')
        tic = time.time()
        audio = decode_audio(audio, split_stereo=split_stereo)
        self.add(f"reading_time={time.time()-tic:.2f} sec")
        if not isinstance(audio, tuple):
            audio = [audio]
        self.add(f"audio_duration={len(audio[0])/16000} sec")
        self.add(f"channels={len(audio)}")
    
        for channel in range(len(audio)):
            logging.info(f'TRANSCRIBING channel {channel+1} ...')
            tic = time.time()
            segments, info = self.model.transcribe(audio[channel], **transcribe)
            self.add(f"ch={channel+1} language_predicted={info.language} probability={info.language_probability}")
            for segment in segments:
                if transcribe.get('word_timestamps') is True:
                    for word in segment.words:
                        self.add(f"ch={channel+1} start={word.start:.3f} end={word.end:.3f} txt={word.word}")
                else:
                    self.add(f"ch={channel+1} start={segment.start:.3f} end={segment.end:.3f} txt={segment.text}")
            self.add(f"ch={channel+1} inference_time={time.time()-tic:.2f} sec")
        logging.info(f'DONE')
        return self.out

    def add(self, line):
        if self.fdo is not None:
            self.fdo.write(line + '\n')
        self.out.append(line)
