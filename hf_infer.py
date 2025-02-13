from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import time
import logging
import numpy as np
from scipy.io import wavfile

class infer():

    def __init__(self, model, load={}):
        logging.info(f'model={model}')
        logging.info(f"load={load}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperForConditionalGeneration.from_pretrained(model).to(device)
        self.processor = WhisperProcessor.from_pretrained(model)
        logging.info(f"model loaded")

    def audio_file(self, audio, split_stereo=False):
        # audio is either a string containing the path to an audio file or a numpy.ndarray resulting from decode_audio
        audio, sr = torchaudio.load(audio)
        self.audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
        logging.info(f"audio.shape={self.audio.shape}")
        if len(self.audio) == 1:
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
            if channel != ch:
                continue
            tic = time.time()
            nwords = 0
            inputs = self.processor(self.audio[ch][start:end], sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            input_features = inputs.input_features.to("cuda")
            attention_mask = inputs.attention_mask.to("cuda")
            forced_decoder_ids = self.processor.tokenizer.get_decoder_prompt_ids(language="fr", task="transcribe")
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, attention_mask=attention_mask, forced_decoder_ids=forced_decoder_ids)
            hyp = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            res.append({'ch': ch+1, 'start': start, 'end': end, 'txt': hyp})
            logging.warning(f"segment audio={(end-start)/16000.0:.3f} sec, time={time.time()-tic:.3f} sec, hyp={hyp}")
    
        if save is not None:
            print(f'save into {save}')
            wavfile.write(save, 16000, np.int16(self.audio[channel][start:end] * 32767))

        return res
