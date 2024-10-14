import argparse
from scripts.utils import align_hyp_to_ref

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Script to align at the segment level asr transcriptions with pakita reference annotations.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ref', type=str, required=True, help='Reference transcription (pakita corpus).')
    parser.add_argument('--ch', type=int, required=True, default='Audio channel of reference file. Either 1 or 2.')
    parser.add_argument('--hyp', type=str, required=True, help='Transcription hypotheses (corresponding to given reference and channel).')
    parser.add_argument('--out', type=str, required=True, help='Output file with aligned hypotheses.')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    h2r = align_hyp_to_ref()
    hyp, ref = h2r(args.hyp, args.ref, args.ch, verbose=args.verbose)

    with open(args.out+'.hyp', 'w') as fd:
        for l in hyp:
            fd.write(l.strip()+'\n')
    
    with open(args.out+'.ref', 'w') as fd:
        for l in ref:
            fd.write(l.strip()+'\n')
        

