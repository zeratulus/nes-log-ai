# NES AI Log Analyzer

This is simple log analyzer enhanced with AI, by default it configured to use Ollama inference with Qwen3-Coder 30b version.
As option, you can use cloud OpenAI ChatGPT models or any other installed Ollama AI model, see Configuration section below.

Currently, this script can process any log file (default cli calls) for any programming language that supports Qwen3-Coder: 
Python, Java, JavaScript, C++, C#, PHP, Go, Ruby, Swift, Kotlin, Rust, TypeScript, SQL, R, Scala, Perl, MATLAB, Lua, 
Dart, Assembly and others (this list of languages I got from Qwen3-Coder chat directly)

With --oc True flag this script designed to process OpenCart log files and log files for my CMS OpenCart capable 
fork called NES (Ninja eCommerce Solution) our pet project here: https://pollyart.store 

PS. This script eats around 40Gb of RAM and around 10Gb of VRAM on my PC with CUDA enabled for more efficient inference,
with only CPU inference it will process your log files slower.

PSPS. To run this experiment with less RAM you can try to use more lightweight model for example: **gemma3n:e4b**

## Installation

1. Install project dependencies
> pip install -r requirements.txt

2. Install Ollama
- Here is link for Windows, macOS and Linux:

> https://ollama.com/download

- Linux cli command:
> curl -fsSL https://ollama.com/install.sh | sh

3. Install AI model for inference **qwen3-coder:30b** in Ollama
> ollama pull qwen3-coder:30b

4. Copy .env.example file and rename it to .env
5. Configure .env for your own purposes, see following section **Configuring environment**
 
## Configuring environment

1. Define project directory DIR_ROOT, temporary I need it for some development purposes. 
It is a directory that contains main.py file. For example: /home/user/projects/nes-log-ai/main.py
> DIR_ROOT="/home/user/projects/nes-log-ai/"
2. IS_LOCAL_OLLAMA_PREFERRED=True flag that indicates to use Ollama if True and ChatGPT if False
>IS_LOCAL_OLLAMA_PREFERRED=True
3. IS_LOCAL_OLLAMA_MAX_PERFORMANCE=True is to use more CPU power by Ollama
> IS_LOCAL_OLLAMA_MAX_PERFORMANCE=True
4. LOCAL_OLLAMA_MODEL_CODER=qwen3-coder:30b variable to set your Ollama AI model, also you can try main.py --model cli 
argument to set any of Ollama AI model to work with.
> LOCAL_OLLAMA_MODEL_CODER=qwen3-coder:30b
5. LOCAL_OLLAMA_MODEL_CODER_NUM_CTX=256000 variable to set current input context window for AI model (maximum for 
qwen3-coder:30b is 256K tokens) if you have some problems with script run, you can try to decrease this number. 
Also take in mind that different AI models has different input token windows, so read model docs on https://ollama.com or on Hugging Face.
> LOCAL_OLLAMA_MODEL_CODER_NUM_CTX=256000
6. LOCAL_OLLAMA_NUM_PREDICT=-1 variable to set maximum AI model output. -1 means that AI model will generate maximally detailed and long possible output.
> LOCAL_OLLAMA_NUM_PREDICT=-1
7. LOCAL_OLLAMA_MODEL_DEFAULT_NUM_CTX=32000 any model that will be passed with --model argument will use this input context window size.
> LOCAL_OLLAMA_MODEL_DEFAULT_NUM_CTX=32000
8. OPENAI_MODEL=gpt-4o ChatGPT model family to use for inference in cloud, works only if IS_LOCAL_OLLAMA_PREFERRED=False
> OPENAI_MODEL=gpt-4o
9. OPENAI_API_KEY=<SOME_API_KEY> insert your OpenAI Api Key here for cloud inference, you can take it here: https://platform.openai.com/
> OPENAI_API_KEY=<SOME_API_KEY>
10. You can configure LangSmith for your installation https://www.langchain.com/langsmith:
>LANGSMITH_TRACING="true"
> 
>LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
> 
>LANGSMITH_API_KEY="<YOUR_LANGSMITH_API_KEY>"
> 
> LANGSMITH_PROJECT="nes-ai-log-analyzer"

## Usage examples

This is example of calling for any log file with English response: 
> python main.py --log /some_path_to_project/nes-log-ai/example/not_nes_error_log

This is example of calling for NES/OpenCart log file with Ukrainian response: 
> python main.py --log /some_path_to_project/nes-log-ai/example/error_log_small --lang uk --oc True

Use double quotes if your log path contains spaces:
> python main.py --log "/some path to project/nes-log-ai/example/not_nes_error_log"