import os
import re
import sys
from tqdm import tqdm
from json import loads
import logging
import argparse
from infer import infer
from utils import ref2list, jiwer_wrap
#from faster_whisper import WhisperModel, decode_audio

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to run inference on Whisper models. Ex: python whisper-infer.py small --audio example.wav --load '{\"device\": \"default\", \"compute_type\": \"default\"}' --transcribe '{\"language\": \"fr\", \"vad_filter\": false}'")
    parser.add_argument('model', type=str, help='Path or name of the Whisper model. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/utils.py#L12)')
    parser.add_argument('--ids', type=str, required=True, help='File with list of audio files to evaluate.')    
    parser.add_argument('--output', type=str, default=None, help='Output path for [output].(ref,hyp) transcriptions.')    
    parser.add_argument('--load', type=str, default=None, required=False, help="JSON dictionary to control model loading. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/transcribe.py#L583)")
    parser.add_argument('--transcribe', type=str, default=None, required=False, help="JSON dictionary to control transcription. (https://github.com/SYSTRAN/faster-whisper/blob/d57c5b40b06e59ec44240d93485a95799548af50/faster_whisper/transcribe.py#L705)")
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()
    flog = None if args.output is None else args.output+'.log'
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO' if args.verbose else 'WARNING'), filename=flog)

    sampling_rate = 16000
    prefix_wav = 'corpus/pakita/fre/cts/data/audio/'
    prefix_trs = 'corpus/pakita/fre/cts/data/trans/deta/stm/'

    with open(args.ids, 'r') as fdi:
        names = [l.strip() for l in fdi]
    references = [prefix_trs+name+'.stm' for name in names]
    audios = [prefix_wav+name+'.wav' for name in names]

    load = loads(args.load) if args.load is not None else {}
    transcribe = loads(args.transcribe) if args.transcribe is not None else {}
    s = jiwer_wrap(uppercase=True, no_punct=True, no_hesit=True, no_noise=True, split_apos=True)

    if args.output is not None:
        fdh = open(args.output+'.hyp', 'w')
        fdr = open(args.output+'.ref', 'w')
        fde = open(args.output+'.err', 'w')

    pattern_noise = r"^\[[^\s]*\]$"
    refs, hyps = [], []
    w = infer(args.model, load=load)
    for i in range(len(audios)):
        w.audio_file(audios[i], split_stereo=True)
        for ch in range(2):
            ref = ref2list(references[i], channel=ch)
            for k in tqdm(range(len(ref)), desc=f'channel:{ch}'):
                txt = ref[k]['txt']
                if txt.startswith('[') and txt.endswith(']') and ' ' not in txt:
                    logging.info(f'skipping ref transcription: \'{txt}\'')
                    continue
                refs.append(txt)
                res = w(channel=ch, start=ref[k]['start'], end=ref[k]['end'], transcribe=transcribe)
                hyps.append(''.join([r['txt'] for r in res]))
                wer = s(refs[-1], hyps[-1])
                if args.output is not None:
                    fdh.write(f"{hyps[-1]}\n")
                    fdh.flush()
                    fdr.write(f"{refs[-1]}\n")
                    fdr.flush()
                    fde.write(f"{names[i]}:{ch}:{ref[k]['start']}-{ref[k]['end']} {wer}\n")
                    fde.flush()
            mywer = s(refs, hyps)
            fde.write(f"*** {names[i]}:{ch}: {mywer}\n")
            fde.flush()
            print(mywer)

    if args.output is not None:
        fdh.close()
        fdr.close()
        fde.close()
