import re
import sys
import jiwer

#pattern_hyp = r'\[(\d+\.\d+) -> (\d+\.\d+)\] (.+)' #[2.400 -> 2.920] allô?
pattern_ref = r'PKCTS\S* (\d+) PKCTS\S* (\d+\.\d+) (\d+\.\d+) \S+ (.+)' #PKCTS01_FRE_FR_00186_02 1 PKCTS01_FRE_FR_00186_02_spk1 2.325 3.075 <o,f0,male> allô David ? 
pattern_hyp = r'ch=(\d) start=(\d+\.\d+) end=(\d+\.\d+) txt=(.+)'

def ref2list(file, channel):
    entries = []
    with open(file, 'r') as fd:
        for l in fd: 
            l = l.strip()
            match = re.match(pattern_ref, l)
            if match:
                ch = int(match.group(1))-1
                if ch != channel:
                    continue
                start = float(match.group(2))
                end = float(match.group(3))
                txt = match.group(4)
                entries.append({'start': start, 'end': end, 'txt': txt})
    return entries

def hyp2list(file, channel):
    entries = []
    with open(file, 'r') as fd:
        for l in fd:
            match = re.match(pattern_hyp, l.strip())
            if match:
                ch = int(match.group(1))
                if ch != channel:
                    continue
                start = float(match.group(2))
                end = float(match.group(3))
                txt = match.group(4)
                entries.append({'start': start, 'end': end, 'txt': txt})
    return entries

def file2list(file, input_type='raw'):
    segments = []
    with open(file, 'r') as fd:
        for l in fd:
            segments.append(l.strip())
    return segments


class jiwer_wrap():
    def __init__(self, show_alignments=False, skip_correct=False, use_words=False, uppercase=False, no_punct=False, no_hesit=False, no_noise=False, split_apos=False, single_line=False):
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

        if self.single_line:
            hyp = [''.join(hyp)]
            ref = [' '.join(ref)]

        hyp = [self.preprocess(h, False) for h in hyp]
        ref = [self.preprocess(r, True) for r in ref]
            
        if self.show_alignments:
            res = jiwer.process_words(ref, hyp) if self.use_words else jiwer.process_characters(ref, hyp)
            print(jiwer.visualize_alignment(res, show_measures=False, skip_correct=self.skip_correct))
            
        measures = jiwer.compute_measures(ref, hyp)
        #print(measures.keys())
        wer = 100*measures['wer']
        D = measures['deletions']
        S = measures['substitutions']
        I = measures['insertions']
        H = measures['hits']
        N = sum([len(r) for r in measures['truth']])
        return f"WER {wer:.2f} err {D+S+I} len {N} sub {S} del {D} ins {I} hit {H}"


class align_hyp_to_ref:

    def __init__(self):
        pass

    def align_hyp_up_to(self, end, verbose=False):
        txt = []
        while len(self.hyp):
            s, e, t = self.hyp[0]['start'], self.hyp[0]['end'], self.hyp[0]['txt']
            if verbose:
                print(f'\tword [{s}, {e}) {t}')
            if len(self.ref)==0 or s < end:
                txt.append(t)
                self.hyp.pop(0)
                if verbose:
                    print(f'\tadded')
            else:
                if verbose:
                    print(f'\tbreak')
                break
        if len(txt) == 0:
            txt.append('<emptyHyp>')
        return ''.join(txt).strip()
    
    def __call__(self, fhyp, fref, channel, verbose=False):
        self.ref = ref2list(fref, channel)
        self.hyp = hyp2list(fhyp, channel)
        REF = []
        HYP = []    
        hyp_line = []
        while len(self.ref):
            curr_ref = self.ref.pop(0)
            start, end, txt = curr_ref['start'], curr_ref['end'], curr_ref['txt']
            REF.append(txt)
            if verbose:
                print(f'REF: [{start}, {end}) {txt}')
            txt = self.align_hyp_up_to(end, verbose)
            HYP.append(txt)
            if verbose:
                print(f'HYP: {txt}')

        print(f'Built lists with HYP={len(HYP)} REF={len(REF)} segments')
        assert len(HYP) == len(REF)
        return HYP, REF
        
