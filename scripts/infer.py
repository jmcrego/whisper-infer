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

    def __call__(self, audio, split_stereo=False, transcribe={}):
        self.audio_name = audio
        self.split_stereo = split_stereo
        self.transcribe = transcribe
        logging.info(f'READING {audio} ...')
        tic = time.time()
        self.audio = decode_audio(audio, split_stereo=split_stereo)
        if not isinstance(self.audio, tuple):
            audio = [self.audio]

    def __iter__(self):
        yield f"model={self.model_name}"
        yield f"audio={self.audio_name}"
        yield f"split_stereo={self.split_stereo}"
        yield f"load={self.load}"
        yield f"transcribe={self.transcribe}"
        yield f"audio_duration={len(self.audio[0])/16000} sec"
        yield f"channels={len(self.audio)}"
        for channel in range(len(self.audio)):
            logging.info(f'TRANSCRIBING channel {channel+1} ...')
            tic = time.time()
            segments, info = self.model.transcribe(self.audio[channel], **self.transcribe)
            yield f"ch={channel+1} language_predicted={info.language} probability={info.language_probability}"
            for segment in segments:
                if self.transcribe.get('word_timestamps') is True:
                    for word in segment.words:
                        yield f"ch={channel+1} start={word.start:.3f} end={word.end:.3f} txt={word.word}"
                else:
                    yield f"ch={channel+1} start={segment.start:.3f} end={segment.end:.3f} txt={segment.text}"
            yield f"ch={channel+1} inference_time={time.time()-tic:.2f} sec"
        logging.info(f'DONE')


