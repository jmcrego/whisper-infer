import re
import jiwer

def file2list(file):
    segments = []
    with open(file, 'r') as fd:
        for l in fd:
            segments.append(l.strip())
    print(f'found {len(segments)} segments in {file})', file=sys.stderr)
    return segments


class jiwer_wrap():
    def __init__(self, show_measures=True, show_alignments=False, skip_correct=False, use_words=True, uppercase=False, no_punct=False, no_hesit=False, no_noise=False, split_apos=False, single_line=False):
        self.show_measures=show_measures
        self.show_alignments=show_alignments
        self.skip_correct=skip_correct
        self.use_words = use_words
        self.uppercase = uppercase
        self.no_punct = no_punct
        self.no_hesit = no_hesit
        self.no_noise = no_noise
        self.split_apos = split_apos
        self.single_line = single_line
        
        self.noise = r"\[[^\s\]]+\]"
        self.hesit = r"\b(euh|beuh|hm|hein)\b"
        self.punct = r"([,.!?;:\)\(\]\[\{\}\*&-])"
        self.punct_del = r"([\)\(])"
        self.apost = r"(\w+)'(\w+)"
        #self.chars = r"[\(\)\*&]"
        #self.initp = r"^\s*[,.!?;:]\s+"
        self.spaces = r"\s\s+"

        
    def preprocess(self, txt, isRef):
        if self.no_noise:
            txt = re.sub(self.noise, '', txt)
        if self.split_apos:
            txt = re.sub(self.apost, r"\1' \2", txt)
        if self.no_hesit:
            txt = re.sub(self.hesit, '', txt)
        if self.no_punct:
            txt = re.sub(self.punct, ' ', txt)
            txt = re.sub(self.punct_del, '', txt)
        if self.uppercase:
            txt = txt.upper()
        txt = re.sub(self.spaces, ' ', txt.strip())
        if txt == '' and isRef:
            txt = '<emptyRef>'
        return txt
        
    def __call__(self, hyp, ref):
        if isinstance(hyp, str):
            hyp = [hyp]
        if isinstance(ref, str):
            ref = [ref]
        hyp = [self.preprocess(h, False) for h in hyp]
        ref = [self.preprocess(r, True) for r in ref]

        if self.single_line:
            hyp = [' '.join(hyp)]
            ref = [' '.join(ref)]
            
        if self.show_alignments:
            res = jiwer.process_words(ref, hyp) if self.use_words else jiwer.process_characters(ref, hyp)
            print(jiwer.visualize_alignment(res, show_measures=False, skip_correct=self.skip_correct))
            
        if self.show_measures:
            wer = jiwer.wer(ref, hyp)
            cer = jiwer.cer(ref, hyp)
            print(f'wer={wer}')
            print(f'cer={cer}')
        


