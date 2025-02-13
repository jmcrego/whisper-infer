#This makes each class accessible when importing scripts as a package.
from .infer import infer
from .utils import ref2list
from .utils import hyp2list
from .utils import file2list
from .utils import align_hyp_to_ref
from .utils import jiwer_wrap

#The __all__ list explicitly defines what will be imported when using: from scripts import *
__all__ = ["infer", "ref2list", "hyp2list", "file2list", "align_hyp_to_ref", "jiwer_wrap"]
