import sys
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

def main():
    model_path = "/Users/minerootsios/Downloads/shiv/models/Qwen2.5-VL-7B-Instruct-Q6_K.gguf"
    mmproj_path = "/Users/minerootsios/Downloads/shiv/models/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"
    
    print("Loading mmproj...")
    try:
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
    except Exception as e:
        print(f"Error loading chat handler: {e}")
        return
        
    print("Loading model...")
    try:
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=2048,
            n_gpu_layers=-1 # Use Metal on Mac
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print("Model loaded successfully!")
    
if __name__ == "__main__":
    main()
