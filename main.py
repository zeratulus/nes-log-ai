from datetime import datetime
import argparse
import os
import logging
import dotenv
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from nes.apache_php_log_parser import format_error_item_to_str, save_json_file
from nes.langchain_helpers import ollama_response_to_dict
from nes.functions import init_llm, torch_info, simple_error_detector
import nes.apache_php_log_parser as parser
from nes.log_ai_processor import LogAiProcessor
import qdrant_client
from nes.qdrant.fastembed_functions import get_dense_model_vector_size, get_late_interaction_model_vector_size
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from nes.qdrant.qdrant_hybrid_search import QdrantHybridSearchClient, QDRANT_EMB_DENSE_MODEL_NAME, QDRANT_EMB_SPARSE_MODEL_NAME, QDRANT_EMB_LATE_ITER_MODEL_NAME

dotenv.load_dotenv()

DIR_CURRENT = os.getenv("DIR_ROOT")
DIR_LOGS = os.environ.get('DIR_LOGS')
os.makedirs(DIR_LOGS, exist_ok=True)

torch_info()

current_log_level = logging.INFO
IS_DEBUG = False
if bool(os.environ.get("IS_DEBUG")):
    IS_DEBUG = True

if IS_DEBUG:
     current_log_level = logging.DEBUG

now = datetime.now()
log_file = f"{DIR_LOGS}{now.strftime(format='%Y%m%d-%H%M%S')}.log"
logging.basicConfig(level=current_log_level,
                    format='[%(asctime)s][%(levelname)s] - %(message)s',
                    filename=log_file,
                    filemode='a'
                    )

args = argparse.ArgumentParser()
args.add_argument('--log', type=str, dest='log_file', required=True, help='Path to log file to process with AI')
args.add_argument('--lang', type=str, dest='language', default="en", required=False, help='Current processing language, available: uk, en')
args.add_argument('--model', type=str, dest='model', default="", required=False, help='LLM Model for log processing')
args.add_argument('--oc', type=bool, dest='is_nes_parsing', default=False, required=False, help='Parse with economical NES/OpenCart log processing, use it for NES/Opencart logs')
args = args.parse_args()

log_file_path = args.log_file
if not os.path.isfile(log_file_path):
    raise FileNotFoundError(f"Provided log file {log_file_path} not found")

log_path, log_file_name = os.path.split(log_file_path)

json_file_name = DIR_CURRENT + f"/outputs/{now.strftime(format=f"{log_file_name}-%Y%m%d-%H%M%S")}.json"
DIR_OUTPUTS = os.path.dirname(json_file_name) + f"/{log_file_name}/{now.strftime(format=f"%Y%m%d-%H%M%S")}/"
os.makedirs(DIR_OUTPUTS, exist_ok=True)

parsed_data = None
if args.is_nes_parsing:
    parsed_data = parser.parse_log_file(log_file_path)
    if not parsed_data:
        raise Exception(f"Provided log file {log_file_path} do not contains NES/Opencart structure")

CURRENT_LLM_MODEL = os.environ.get('LOCAL_OLLAMA_MODEL_CODER')
CURRENT_LLM_NUM_CTX = int(os.environ.get('LOCAL_OLLAMA_MODEL_CODER_NUM_CTX'))
if args.model != "":
    CURRENT_LLM_MODEL = args.model
    CURRENT_LLM_NUM_CTX = int(os.environ.get('LOCAL_OLLAMA_MODEL_DEFAULT_NUM_CTX'))

llm = init_llm(CURRENT_LLM_MODEL, CURRENT_LLM_NUM_CTX)

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION_NAME")
QDRANT_COLLECTION_NAME = f"{QDRANT_COLLECTION}"
try:
    qdrant_client_instance = qdrant_client.QdrantClient(url=QDRANT_URL, prefer_grpc=False)
except Exception as e:
    print(f"Помилка підключення до Qdrant за адресою {QDRANT_URL}.")
    print("Переконайтеся, що ви запустили Qdrant")
    raise e

qdrant_hybrid = QdrantHybridSearchClient(client=qdrant_client_instance, collection_name=QDRANT_COLLECTION_NAME)

logging.info(f"Створення нової колекції '{QDRANT_COLLECTION_NAME}' з гібридною конфігурацією...")
qdrant_hybrid.create_collection_if_not_exists(collection_name=QDRANT_COLLECTION_NAME)
logging.info("Колекцію успішно створено.")

logging.info("Ініціалізація моделей ембедингів з fastembed...")
try:
    dense_embedding_model = TextEmbedding(QDRANT_EMB_DENSE_MODEL_NAME)
    sparse_embedding_model = SparseTextEmbedding(QDRANT_EMB_SPARSE_MODEL_NAME)
    late_interaction_embedding_model = LateInteractionTextEmbedding(QDRANT_EMB_LATE_ITER_MODEL_NAME)
    logging.info("Моделі ембедингів успішно ініціалізовано.")
except Exception as e:
    logging.error(f"Не вдалося ініціалізувати моделі fastembed: {e}")

processor = LogAiProcessor(llm=llm, parsed_data=parsed_data, args=args, outputs_dir=DIR_OUTPUTS, json_file_name=json_file_name, qdrant_hybrid=qdrant_hybrid, log_file_name=log_file_name)

#Processing for NES/Opencart log files
if parsed_data and args.is_nes_parsing:
    processor.process_opencart_logs()

#Processing for any log type
if not args.is_nes_parsing:
    processor.process_logs(log_file_path=log_file_path)