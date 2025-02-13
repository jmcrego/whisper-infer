import sys
import time
import logging
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel, decode_audio

class infer():

    def __init__(self, model, load={}):
        # model is either a Systran model (medium, small, ...) or the path to a local checkpoint
        logging.info(f'model={model}')
        logging.info(f"load={load}")
        self.model = WhisperModel(model, **load)
        logging.info(f"model loaded")

    def audio_file(self, audio, split_stereo=False):
        # audio is either a string containing the path to an audio file or a numpy.ndarray resulting from decode_audio
        self.audio = decode_audio(audio, split_stereo=split_stereo) if isinstance(audio, str) else audio
        if not isinstance(self.audio, tuple):
            self.audio = [self.audio]
        logging.info(f"audio duration={len(self.audio[0])/16000} sec")
        logging.info(f"audio channels={len(self.audio)}")

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
            curr_res = []
            if channel != ch:
                continue
            tic = time.time()
            nwords = 0
            segments, info = self.model.transcribe(self.audio[ch][start:end], **transcribe)
            for segment in segments:
                if transcribe.get('word_timestamps') is True:
                    for word in segment.words:
                        curr_res.append({'ch': ch+1, 'start': word.start, 'end': word.end, 'txt': word.word})
                        nwords += 1
                else:
                    curr_res.append({'ch': ch+1, 'start': segment.start, 'end': segment.end, 'txt': segment.text})
                    nwords += len(segment.text.split(' '))
            hyp=''.join([e['txt'] for e in curr_res])
            logging.warning(f"segment audio={(end-start)/16000.0:.3f} sec, time={time.time()-tic:.3f} sec, hyp={hyp}")
            res += curr_res

        if save is not None:
            print(f'save into {save}')
            wavfile.write(save, 16000, np.int16(self.audio[channel][start:end] * 32767))

        return res
