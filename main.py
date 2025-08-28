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
args.add_argument('--not-nes', type=bool, dest='skip_nes_parsing', default=False, required=False, help='Skip the parsing step of economical NES/OpenCart log processing, use it for non NES/Opencart logs')
args = args.parse_args()

log_file_path = args.log_file
if not os.path.isfile(log_file_path):
    raise FileNotFoundError(f"Provided log file {log_file_path} not found")

log_path, log_file_name = os.path.split(log_file_path)

json_file_name = DIR_CURRENT + f"/outputs/{now.strftime(format=f"{log_file_name}-%Y%m%d-%H%M%S")}.json"
DIR_OUTPUTS = os.path.dirname(json_file_name) + f"/{log_file_name}/{now.strftime(format=f"%Y%m%d-%H%M%S")}/"
os.makedirs(DIR_OUTPUTS, exist_ok=True)

parsed_data = None
if not args.skip_nes_parsing:
    parsed_data = parser.parse_log_file(log_file_path)
    if not parsed_data:
        raise Exception(f"Provided log file {log_file_path} do not contains NES/Opencart structure")

CURRENT_LLM_MODEL = os.environ.get('LOCAL_OLLAMA_MODEL_CODER')
CURRENT_LLM_NUM_CTX = int(os.environ.get('LOCAL_OLLAMA_MODEL_CODER_NUM_CTX'))
if args.model != "":
    CURRENT_LLM_MODEL = args.model
    CURRENT_LLM_NUM_CTX = int(os.environ.get('LOCAL_OLLAMA_MODEL_DEFAULT_NUM_CTX'))

llm = init_llm(CURRENT_LLM_MODEL, CURRENT_LLM_NUM_CTX)

#Processing for NES/Opencart log files
if parsed_data and not args.skip_nes_parsing:
    parser.print_summary(parsed_data, args.language)
    parser.save_json_file(parsed_data, json_file_name)

    with open(f"prompts/anal-logs-nes-{args.language}.prompt") as f:
        base_template = f.read()

    prompt_template = PromptTemplate(input_variables=["error_details"], template=base_template)

    for log_key, log_obj in tqdm(parsed_data.items()):
        prompt = prompt_template.format_prompt(error_details=format_error_item_to_str(log_key, log_obj, args.language))
        response = llm.invoke(prompt)
        save_json_file(ollama_response_to_dict(response), DIR_OUTPUTS + str(log_key) + ".json")
        with open(DIR_OUTPUTS + str(log_key) + ".txt", 'w') as f:
            f.write(response.content)

#Processing for not NES/Opencart log files
if args.skip_nes_parsing:
    with open(f"prompts/anal-logs-not-nes-{args.language}.prompt") as f:
        base_template = f.read()

    prompt_template = PromptTemplate(input_variables=["error_details"], template=base_template)

    with open(log_file_path, 'r') as f:
        line_idx = 0
        for line in tqdm(f):
            if simple_error_detector(line):
                prompt = prompt_template.format_prompt(error_details=line)
                response = llm.invoke(prompt)
                save_json_file(ollama_response_to_dict(response), DIR_OUTPUTS + str(line_idx) + ".json")
                with open(DIR_OUTPUTS + str(line_idx) + ".txt", 'w') as fs:
                    fs.write(response.content)

            line_idx += 1