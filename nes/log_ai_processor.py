from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from nes.apache_php_log_parser import format_error_item_to_str, save_json_file
import nes.apache_php_log_parser as parser
from nes.functions import simple_error_detector
from nes.langchain_helpers import ollama_response_to_dict


class LogAiProcessor(object):

    def __init__(self, llm, parsed_data, args, outputs_dir: str, json_file_name: str):
        self.llm = llm
        self.parsed_data = parsed_data
        self.args = args
        self.outputs_dir = outputs_dir
        self.json_file_name = json_file_name

    def process_opencart_logs(self):
        if self.parsed_data and self.args.is_nes_parsing:
            parser.print_summary(self.parsed_data, self.args.language)
            parser.save_json_file(self.parsed_data, self.json_file_name)

            with open(f"prompts/anal-logs-nes-{self.args.language}.prompt") as f:
                base_template = f.read()

            prompt_template = PromptTemplate(input_variables=["error_details"], template=base_template)

            for log_key, log_obj in tqdm(self.parsed_data.items()):
                prompt = prompt_template.format_prompt(error_details=format_error_item_to_str(log_key, log_obj, self.args.language))
                response = self.llm.invoke(prompt)
                save_json_file(ollama_response_to_dict(response), self.outputs_dir + str(log_key) + ".json")
                with open(self.outputs_dir + str(log_key) + ".txt", 'w') as f:
                    f.write(response.content)

    #Processing for not NES/Opencart log files
    def process_logs(self, log_file_path: str):
        if not self.args.is_nes_parsing:
            with open(f"prompts/anal-logs-not-nes-{self.args.language}.prompt") as f:
                base_template = f.read()
    
            prompt_template = PromptTemplate(input_variables=["error_details"], template=base_template)
    
            with open(log_file_path, 'r') as f:
                line_idx = 0
                for line in tqdm(f):
                    if simple_error_detector(line):
                        prompt = prompt_template.format_prompt(error_details=line)
                        response = self.llm.invoke(prompt)
                        save_json_file(ollama_response_to_dict(response), self.outputs_dir + str(line_idx) + ".json")
                        with open(self.outputs_dir + str(line_idx) + ".txt", 'w') as fs:
                            fs.write(response.content)
    
                    line_idx += 1