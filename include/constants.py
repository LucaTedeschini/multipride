from pathlib import Path
from datetime import datetime


MODELS = [
    "nickprock/setfit-italian-hate-speech",
    "cardiffnlp/twitter-xlm-roberta-base-hate-spanish",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "dccuchile/bert-base-spanish-wwm-uncased",
    "Twitter/twhin-bert-base",
]
LOGS_DIR = Path("./logs").absolute()
OUTPUT_DIR = Path("./output").absolute()
RESULTS_DIR = Path("./results").absolute()
NOW = datetime.now().strftime("%Y%m%d_%H%M%S")