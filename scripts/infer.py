import time
import logging
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel, decode_audio

class infer():

    def __init__(self, model, audio, split_stereo=False, load={}):
        # model is either a Systran model (medium, small, ...) or the path to a local checkpoint
        # audio is either a string containing the path to an audio file or a numpy.ndarray resulting from decode_audio
        logging.info(f'model={model}')
        logging.info(f"load={load}")
        self.model = WhisperModel(model, **load)
        self.audio = decode_audio(audio, split_stereo=split_stereo) if isinstance(audio, str) else audio
        if not isinstance(self.audio, tuple):
            self.audio = [self.audio]
        logging.info(f"channels={len(self.audio)}")
        logging.info(f"duration={len(self.audio[0])/16000} sec")

    def __call__(self, channel=None, start=None, end=None, transcribe={}, save=None):
        logging.info(f"channel={channel}")
        logging.info(f"start={start}")
        logging.info(f"end={end}")
        logging.info(f"transcribe={transcribe}")
        channel = 0 if channel is None else channel
        start = 0 if start is None else int(start*16000)
        end = len(self.audio[0]) if end is None else int(end*16000)
        res = []
        for ch in range(len(self.audio)):
            if channel != ch:
                continue
            segments, info = self.model.transcribe(self.audio[ch][start:end], **transcribe)
            for segment in segments:
                if transcribe.get('word_timestamps') is True:
                    for word in segment.words:
                        res.append({'ch': ch+1, 'start': word.start, 'end': word.end, 'txt': word.word})
                else:
                    res.append({'ch': ch+1, 'start': segment.start, 'end': segment.end, 'txt': segment.text})
        if save is not None:
            print(f'save into {save}')
            wavfile.write(save, 16000, np.int16(self.audio[channel][start:end] * 32767))

        logging.info(f"hyp={''.join([e['txt'] for e in res])}")
        return res

        

