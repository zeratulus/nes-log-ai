import datetime
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from nes.apache_php_log_parser import format_error_item_to_str, save_json_file
import nes.apache_php_log_parser as parser
from nes.functions import simple_error_detector
from nes.langchain_helpers import ollama_response_to_dict
from nes.qdrant.qdrant_hybrid_search import QdrantHybridSearchClient, QDRANT_EMB_DENSE_MODEL_NAME, QDRANT_EMB_SPARSE_MODEL_NAME, QDRANT_EMB_LATE_ITER_MODEL_NAME
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

class LogAiProcessor(object):

    def __init__(self, llm, parsed_data, args, outputs_dir: str, json_file_name: str, qdrant_hybrid: QdrantHybridSearchClient, log_file_name: str):
        self.llm = llm
        self.parsed_data = parsed_data
        self.args = args
        self.outputs_dir = outputs_dir
        self.json_file_name = json_file_name
        self.log_file_name = log_file_name
        self.qdrant_hybrid = qdrant_hybrid
        self.dense_embedding_model = TextEmbedding(QDRANT_EMB_DENSE_MODEL_NAME)
        self.sparse_embedding_model = SparseTextEmbedding(QDRANT_EMB_SPARSE_MODEL_NAME)
        self.late_interaction_embedding_model = LateInteractionTextEmbedding(QDRANT_EMB_LATE_ITER_MODEL_NAME)

    def process_opencart_logs(self):
        if self.parsed_data and self.args.is_nes_parsing:
            parser.print_summary(self.parsed_data, self.args.language)
            parser.save_json_file(self.parsed_data, self.json_file_name)

            with open(f"prompts/anal-logs-nes-{self.args.language}.prompt") as f:
                base_template = f.read()

            prompt_template = PromptTemplate(input_variables=["error_details"], template=base_template)

            collection_name = f"PHP-OpenCart-{self.log_file_name}"
            self.qdrant_hybrid.create_collection_if_not_exists(collection_name)
            self.qdrant_hybrid.collection_name = collection_name

            for log_key, log_obj in tqdm(self.parsed_data.items()):
                error_details = format_error_item_to_str(log_key, log_obj, self.args.language)
                prompt = prompt_template.format_prompt(error_details=error_details)
                response = self.llm.invoke(prompt)
                formatted_response = ollama_response_to_dict(response)
                save_json_file(formatted_response, self.outputs_dir + str(log_key) + ".json")
                with open(self.outputs_dir + str(log_key) + ".txt", 'w') as f:
                    f.write(response.content)

                payload = {
                    "log_record": log_obj,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "llm_response": formatted_response
                }

                # Генерація всіх типів ембедингів для пакету
                dense_embeddings = list(self.dense_embedding_model.passage_embed(log_obj))
                sparse_embeddings = list(self.sparse_embedding_model.passage_embed(log_obj))
                late_interaction_embeddings = list(self.late_interaction_embedding_model.passage_embed(log_obj))

                self.qdrant_hybrid.add_point(doc_id=log_key,
                                             payload=payload,
                                             dense_embedding=dense_embeddings[0].tolist(),
                                             sparse_embedding=sparse_embeddings[0].as_object(),
                                             late_interaction_embedding=late_interaction_embeddings[0].tolist()
                                             )

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
