import os
import threading
import time
import random
import requests
from flask import Flask, jsonify
from openai import OpenAI, APITimeoutError, APIConnectionError, APIError

app = Flask(__name__)

# --- Configuration ---
LITELLM_ENDPOINT = "http://localhost:4000"
API_KEY = "sk-1234" # Replace with your actual key if different
MODELS_TO_USE = ["Qwen/Qwen3-4B", "Qwen/Qwen2.5-7B", "microsoft/Phi-4-mini-instruct", "CohereLabs/c4ai-command-r7b-12-2024", "meta-llama/Llama-3.2-3B-Instruct", "mistralai/Mistral-Nemo-Instruct-FP8-2407"] # Models from models.yaml

# --- Model Specific Configuration ---
# Define EOS tokens for models (add more as needed)
# Check tokenizer_config.json on Hugging Face for the correct token
MODEL_STOP_TOKENS = {
    "Qwen/Qwen3-4B": ["<|im_end|>"],
    "Qwen/Qwen2.5-7B": ["<|im_end|>"],
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ["<|eot_id|>"],
    "meta-llama/Llama-3.2-3B": ["<|eot_id|>"],
    "microsoft/Phi-4-mini-instruct": ["<|eot_id|>"],
    "mistralai/Mistral-Nemo-Instruct-FP8-2407": ["<|eot_id|>"],
    # Example for Llama:
    "meta-llama/Llama-3.2-3B-Instruct": ["<|eot_id|>", "</s>"] # Llama 3 uses <|eot_id|> but check specific tokenizer
    # Example for Mistral:
    # "mistralai/Mistral-7B-Instruct-v0.1": ["</s>"]
}

TRIVIA_QUESTIONS = [
    "What is the capital of France?",
    "Who painted the Mona Lisa?",
    "What is the tallest mountain in the world?",
    "What year did the Titanic sink?",
    "What is the chemical symbol for water?",
    "Which planet is known as the Red Planet?",
    "Who wrote 'Hamlet'?",
    "What is the largest ocean on Earth?",
    "What is the square root of 64?",
    "Who was the first President of the United States?",
    "What was the name of the first general-purpose electronic digital computer?",
    "Who is often considered the first computer programmer for her work on the Analytical Engine?",
    "What does the term 'bug' in computing historically refer to?",
    "What does HTML stand for?",
    "Who invented the World Wide Web?",
    "What operating system kernel was famously created by Linus Torvalds?",
    "What was the name of the Microsoft Office assistant shaped like a paperclip?",
    "What does SQL stand for?",
    "What is the time complexity of an ideal binary search algorithm?",
    "Which popular version control system shares its name with a British insult?",
    "What does 'LASER' stand for?",
    "In networking, what does TCP stand for?",
    "What was the first commercially successful video game?",
    "What is the hexadecimal equivalent of the decimal number 15?",
    "Who co-founded Microsoft with Bill Gates?",
    "Which programming language, known for its logo of a steaming coffee cup, was originally called 'Oak'?",
    "What is the significance of the date January 1, 1970, in Unix systems?",
    "How many programmers does it take to change a light bulb?",
    "What does the 'chmod 777' command typically grant in Unix-like systems?",
    "Before Windows, what was Microsoft's primary text-based operating system?",
    "What observation, often called a 'law', states that the number of transistors on a microchip doubles approximately every two years?",
    "What was the precursor network to the modern Internet, developed by DARPA?",
    "Why don't scientists trust atoms?",
    "What is the answer to the ultimate question of life, the universe, and everything?"
]
DELAY_BETWEEN_QUESTIONS = 3 # Seconds
# DELAY_BETWEEN_MODELS = 2 # Seconds - Removed, handled by DELAY_BETWEEN_QUESTIONS
REQUEST_TIMEOUT_SECONDS = 15 # Timeout for the OpenAI client call

# --- Global State ---
conversation_log = []
trivia_thread = None
stop_thread_flag = threading.Event()

# --- OpenAI Client Initialization ---
# Initialize the client once
client = OpenAI(
    api_key=API_KEY,
    base_url=LITELLM_ENDPOINT, # Point to the LiteLLM proxy
)

# --- Helper Functions ---
def call_litellm(model_name, question):
    """Calls the LiteLLM endpoint using the OpenAI library.
    Returns the answer or an error/status message.
    """
    # Get stop tokens for the model, default to None if not found
    stop_tokens = MODEL_STOP_TOKENS.get(model_name)
    if stop_tokens:
        print(f"DEBUG: Using stop tokens for {model_name}: {stop_tokens}", flush=True)
    else:
        print(f"Warning: No specific stop tokens found for {model_name}. Relying on model default EOS.", flush=True)

    try:
        print(f"DEBUG: Calling OpenAI client for {model_name} with question: {question}", flush=True)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering trivia questions concisely. Answer in a concise manner, do not include any additional information. Use as few tokens as possible."},
                {"role": "user", "content": question}
            ],
            max_tokens=4096,  # ADDED: Set a large safety limit
            stop=stop_tokens, # Use model-specific stop tokens (can be None)
            temperature=0,
            timeout=REQUEST_TIMEOUT_SECONDS # Pass timeout here
        )

        print(f"DEBUG: Raw response object from OpenAI client for {model_name}:\n{response}", flush=True)

        # Access response data using object attributes
        if response.choices:
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            message_content = choice.message.content

            if message_content:
                return message_content.strip()
            else:
                # Content is None or empty, check finish reason
                if finish_reason == 'length':
                    print(f"Warning: Content is null, finish_reason is 'length' for {model_name}.", flush=True)
                    return "(Response truncated by token limit)"
                else:
                    # Log the unexpected finish reason
                    print(f"Warning: Content is null/missing for {model_name} (finish_reason: {finish_reason})", flush=True)
                    return f"(No content returned: {finish_reason})"
        else:
            # Should not happen if API call succeeded, but handle defensively
            print(f"Warning: No choices received from API for {model_name}. Response: {response}", flush=True)
            return "(No response choices received)"

    except APITimeoutError:
        print(f"Error: OpenAI API request timed out after {REQUEST_TIMEOUT_SECONDS}s ({model_name})", flush=True)
        return f"Error: API call timed out ({model_name})"
    except APIConnectionError as e:
        print(f"Error: Failed to connect to OpenAI API (LiteLLM @ {LITELLM_ENDPOINT}). Please check network/proxy. Error: {e}", flush=True)
        return f"Error: Connection to LiteLLM failed ({model_name})"
    except APIError as e:
        # Handle other API errors (e.g., 4xx, 5xx from LiteLLM/vLLM)
        print(f"Error: OpenAI API returned an error: Status={e.status_code}, Response={e.response}, Body={e.body} ({model_name})", flush=True)
        error_detail = str(e.body) if e.body else f"Status {e.status_code}"
        return f"Error: API Error ({model_name}): {error_detail}"
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Error: An unexpected error occurred in call_litellm ({model_name}): {e}", flush=True)
        # Consider logging the full traceback here for debugging
        # import traceback
        # traceback.print_exc()
        return f"Error: Unexpected error ({model_name})"

def run_trivia_game():
    """The background task that runs the trivia game."""
    global conversation_log
    print(">>> Trivia game thread function ENTERED.", flush=True) # ADDED
    # Ensure log is cleared and thread-safe access (though list append is generally safe)
    with app.app_context(): # Ensure access to app context if needed later
        conversation_log.clear()
        conversation_log.append("Trivia game starting...")
    print("Trivia game thread started setup.", flush=True) # MODIFIED

    current_model_index = 0
    random.shuffle(TRIVIA_QUESTIONS) # Shuffle questions each run
    print("Trivia questions shuffled. Entering loop...", flush=True) # ADDED

    for i, question in enumerate(TRIVIA_QUESTIONS):
        print(f"Loop iteration {i}. Checking stop flag...", flush=True) # ADDED
        if stop_thread_flag.is_set():
            with app.app_context():
                conversation_log.append("Trivia game stopped by user.")
            print("Trivia game thread stopping due to flag.", flush=True)
            return

        model_to_ask = MODELS_TO_USE[current_model_index % len(MODELS_TO_USE)]
        
        log_entry_question = f"\n--- Turn {i+1} | Asking '{model_to_ask}' ---"
        log_entry_q_text = f"Q: {question}"
        with app.app_context():
            conversation_log.append(log_entry_question)
            conversation_log.append(log_entry_q_text)
        print(f"Asking {model_to_ask}: {question}", flush=True) # MODIFIED

        answer = call_litellm(model_to_ask, question)
        
        log_entry_answer = f"A: {answer}"
        with app.app_context():
            conversation_log.append(log_entry_answer)
        print(f"Answer from {model_to_ask}: {answer}", flush=True) # MODIFIED

        current_model_index += 1

        # Delay before next question if not the last one
        if i < len(TRIVIA_QUESTIONS) - 1:
            print(f"Sleeping for {DELAY_BETWEEN_QUESTIONS}s", flush=True) # MODIFIED
            # Check flag again during sleep
            stop_requested = stop_thread_flag.wait(timeout=DELAY_BETWEEN_QUESTIONS)
            if stop_requested:
                 with app.app_context():
                     conversation_log.append("Trivia game stopped during delay.")
                 print("Trivia game thread stopping during delay due to flag.", flush=True)
                 return


    with app.app_context():
        conversation_log.append("\n--- Trivia game finished ---")
    print("Trivia game thread finished.", flush=True) # MODIFIED

# --- Flask Routes ---

@app.route('/api/loadgen/start', methods=['POST'])
def start_trivia_api():
    """API endpoint to start the trivia game in a background thread."""
    global trivia_thread
    if trivia_thread and trivia_thread.is_alive():
        print("Start request failed: Trivia already running.", flush=True) # MODIFIED
        return jsonify({"status": "Trivia already running"}), 409 # Conflict

    stop_thread_flag.clear() # Reset the stop flag before starting
    trivia_thread = threading.Thread(target=run_trivia_game, daemon=True)
    trivia_thread.start()
    print("Start request successful, trivia thread started.", flush=True) # MODIFIED
    return jsonify({"status": "Trivia started"})

@app.route('/api/loadgen/stop', methods=['POST'])
def stop_trivia_api():
    """API endpoint to signal the background thread to stop."""
    global trivia_thread
    if not trivia_thread or not trivia_thread.is_alive():
         print("Stop request failed: Trivia not running.", flush=True) # MODIFIED
         return jsonify({"status": "Trivia not running"}), 400 # Bad Request

    stop_thread_flag.set() # Signal the thread to stop
    print("Stop signal sent to trivia thread.", flush=True) # MODIFIED
    # Don't wait (join) here, let the thread stop cleanly in the background
    # The UI will reflect the stop based on polling the log
    return jsonify({"status": "Stop signal sent"})


@app.route('/api/loadgen/log')
def get_log_api():
    """API endpoint to return the current conversation log."""
    # Return a copy to avoid potential modification issues if any
    with app.app_context():
        log_copy = list(conversation_log)
    return jsonify(log_copy)

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible within Docker network from Nginx
    # Use port 5002 as planned
    print("Starting Flask Load Generator on 0.0.0.0:5002", flush=True) # MODIFIED
    # Disable Flask's default reloader in production/docker, but keep debug for now
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False) 