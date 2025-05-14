from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # My specific case. Tweak for your folder organization
RESULTS_FOLDER = PROJECT_ROOT / "results"

MODEL = 'model'
PRE_INP = 'preprocess_input'
DEC_PRED = 'decode_predictions'
SHAPE = 'shape'
ICON = 'icon'
SOURCE = 'source'

FILE = 'file'
SIM_CLASSES = 'similar classes (count)'
SIM_CLASSES_PERC = 'similar classes (%)'
SIM_BEST_CLASS = 'similar best class'

MAX_INFO_SAMPLE_SIZE = 50