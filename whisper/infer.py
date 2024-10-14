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
        fdo.write(f"model={self.model_name}" + "\n")
        fdo.write(f"audio={audio}" + "\n")
        fdo.write(f"split_stereo={split_stereo}" + "\n")
        fdo.write(f"load={self.load}" + "\n")
        fdo.write(f"transcribe={transcribe}" + "\n")

        logging.info(f'READING {audio} ...')
        tic = time.time()
        audio = decode_audio(audio, split_stereo=split_stereo)
        fdo.write(f"reading_time={time.time()-tic:.2f} sec" + "\n")
        if not isinstance(audio, tuple):
            audio = [audio]
        fdo.write(f"audio_duration={len(audio[0])/16000} sec" + "\n")
        fdo.write(f"channels={len(audio)}" + "\n")
    
        for channel in range(len(audio)):
            logging.info(f'TRANSCRIBING channel {channel+1} ...')
            tic = time.time()
            segments, info = self.model.transcribe(audio[channel], **transcribe)
            fdo.write(f"ch={channel+1} language_predicted={info.language} probability={info.language_probability}" + "\n")
            for segment in segments:
                if transcribe.get('word_timestamps') is True:
                    for word in segment.words:
                        fdo.write(f"ch={channel+1} start={word.start:.3f} end={word.end:.3f} txt={word.word}" + "\n")
                else:
                    fdo.write(f"ch={channel+1} start={segment.start:.3f} end={segment.end:.3f} txt={segment.text}" + "\n")
            fdo.write(f"ch={channel+1} inference_time={time.time()-tic:.2f} sec" + "\n")
        logging.info(f'DONE')

