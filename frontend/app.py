print("")
print("[+] Initializing libraries used for this program...")
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, logging, BitsAndBytesConfig
logging.set_verbosity_error() # Silence annoying tokenizer warnings
print("    [+] transformer has been imported.")
from accelerate import dispatch_model
print("    [+] accelerate has been imported.")
import customtkinter as ctk
print("    [+] customtkinter has been imported.")
import datetime
print("    [+] datetime has been imported.")
import torch
print("    [+] torch has been imported.")
import time
print("    [+] time has been imported.")
import os
print("    [+] os has been imported.")
print("[+] Initialization has finished.")
print("")

# Ensure the offloading directory exists or create it
offload_dir = "./offload_dir"  # Replace with your desired directory
os.makedirs(offload_dir, exist_ok=True)

# Set bnb config due to large size of models
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    offload_folder=offload_dir  # Try explicitly setting it here as well
)


# Handles picking what model to use for the application
valid_input = False
while not valid_input:
    print("[?] Choose which model you would like to use:")
    print("    1. Our Model (DOES NOT WORK)")
    print("    2. bagasbgs2516's Model (DOES NOT WORK)")
    print("    3. AgriQBot (Works)")
    model_choice = input("> ")
    
    # Handle choices
    if model_choice == "1":
        model = "our_model"
        valid_input = True # Exit loop
    elif model_choice == "2":
        model = "bagasbgs2516"
        valid_input = True # Exit loop
    elif model_choice == "3":
        model = "AgriQBot"
        valid_input = True # Exit loop
    else:
        print("INVALID INPUT. You must enter '1' or '2' or '3'.")
        time.sleep(3)
        for _ in range(20):
            print("")


if model == "our_model":
    print("")
    print("You have chosen to use our model.")
    print("")
    print("~ Information ~")
    print(" - Base Model: mistralai/Mistral-7B-v0.3")
    print(" - Our Model:  czalpha/fine_tuned_model")
    print(" - Tokenizer:  mistralai/Mistral-7B-v0.3")
    print("")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    # Load model with offload_folder
    model = AutoModelForCausalLM.from_pretrained(
        "czalpha/fine_tuned_model",
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder=offload_dir,
        offload_state_dict=True,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.config.use_cache = False
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        max_length=512,
        clean_up_tokenization_spaces=True
    )
elif model == "bagasbgs2516":
    print("")
    print("You have chosen to use bagasbgs2516's model.")
    print("")
    print("~ Information ~")
    print(" - Base Model: meta-llama/Llama-2-7b-hf")
    print(" - bagasbgs2516's Model:  bagasbgs2516/llama2-agriculture-lora")
    print(" - Tokenizer:  bagasbgs2516/llama2-agriculture-lora")
    print("")
    
    tokenizer = AutoTokenizer.from_pretrained("bagasbgs2516/llama2-agriculture-lora")
    model = AutoModelForCausalLM.from_pretrained(
        "bagasbgs2516/llama2-agriculture-lora",
        quantization_config=bnb_config,  # <-- pass the full config
        device_map="auto"
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
elif model == "AgriQBot":
    print("")
    print("You have chosen to use AgriQBot.")
    print("")
    print("~ Information ~")
    print(" - Creator: mrSoul7766")
    print(" - Model Link: https://huggingface.co/mrSoul7766/AgriQBot")
    print(" - Dataset Link: https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only")
    print("")
    # Automatically choose GPU if available
    device = 0 if torch.cuda.is_available() else -1
    print(f"[ ] Loading pipeline to mrSoul7766's AgriQbot model... (Device: {'GPU' if device == 0 else 'CPU'})")
    
    # Load the model using Hugging Face pipeline
    pipe = pipeline(
        "text2text-generation", 
        model="mrSoul7766/AgriQBot", 
        device=device)
    print("[+] Pipeline loaded.", "\n\n")


# App settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Data to store chats
chat_sessions = []
current_chat_index = -1


def get_timestamp():
    """Returns a timestamp string."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def new_chat():
    # Function to start a new chat
    global current_chat_index
    current_chat_index = len(chat_sessions)
    chat_sessions.append([])  # Add empty session
    update_history_buttons()
    clear_chat_display()


def clear_chat_display():
    for widget in chat_scroll_frame.winfo_children():
        widget.destroy()


def send_message():
    global current_chat_index
    user_msg = user_input.get()
    if user_msg.strip() == "":
        return
    
    if current_chat_index == -1:
        new_chat()
    
    chat_sessions[current_chat_index].append(("You", user_msg))
    add_message_bubble("You", user_msg)
    
    # Use the Hugging Face pipeline to generate the AI response
    response = pipe(
        user_msg,
        max_length=200,
        do_sample=True,
        truncation=True,
        pad_token_id=pipe.model.config.eos_token_id  # Explicitly set pad token
    )[0]["generated_text"]
    
    # Try to trim out the prompt if it gets echoed
    if response.lower().startswith(user_msg.lower()):
        response = response[len(user_msg):].strip()
    
    chat_sessions[current_chat_index].append(("AI", response))
    add_message_bubble("AI", response)
    
    user_input.delete(0, "end")


# Display message bubble
def add_message_bubble(sender, message):
    bubble_color = "#1f273f" if sender == "You" else "#293046"
    anchor_side = "e" if sender == "You" else "w"
    text_color = "white"

    bubble = ctk.CTkLabel(
        chat_scroll_frame,
        text=message,
        text_color=text_color,
        fg_color=bubble_color,
        corner_radius=15,
        wraplength=350,
        justify="left",
        padx=10,
        pady=4
    )
    bubble.pack(anchor=anchor_side, pady=4, padx=10)


# Load previous chat session
def load_chat(index):
    global current_chat_index
    current_chat_index = index
    clear_chat_display()
    for sender, msg in chat_sessions[index]:
        add_message_bubble(sender, msg)


# Create chat history buttons
def update_history_buttons():
    for widget in history_frame.winfo_children():
        widget.destroy()
    for i, session in enumerate(chat_sessions):
        label = f"Chat {i + 1}"
        btn = ctk.CTkButton(history_frame, text=label, width=200, height=30,
                            command=lambda idx=i: load_chat(idx),
                            fg_color="#1f273f", hover_color="#293046")
        btn.pack(pady=5, padx=10)


# Main window
window = ctk.CTk()
window.title("AI Messaging App")
window.geometry("800x500")
window.configure(fg_color="#414860")

window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=3)
window.grid_rowconfigure(0, weight=1)

# ==== SIDE PANEL ====
side_panel = ctk.CTkFrame(window, width=250, fg_color="#2e344f", corner_radius=0)
side_panel.grid(row=0, column=0, sticky="nsew")
side_panel.grid_propagate(False)

# Header with History title + small "+" button
header_frame = ctk.CTkFrame(side_panel, fg_color="#2e344f")
header_frame.pack(fill="x", pady=(10, 5), padx=10)

history_label = ctk.CTkLabel(
    header_frame,
    text="History",
    font=ctk.CTkFont(size=16, weight="bold"),
    text_color="white"
)
history_label.pack(side="left", padx=(40, 5))

small_plus_button = ctk.CTkButton(
    header_frame,
    text="+",
    width=30,
    height=30,
    command=new_chat,
    corner_radius=8,
    fg_color="#293046",
    hover_color="#1e2333",
    font=ctk.CTkFont(size=14, weight="bold")
)
small_plus_button.pack(side="right")

# Scrollable chat history list
history_frame = ctk.CTkScrollableFrame(side_panel, fg_color="#2e344f", width=220, height=400)
history_frame.pack(fill="both", expand=True, padx=10, pady=5)

# ==== MAIN CHAT AREA ====
main_area = ctk.CTkFrame(window, fg_color="#414860")
main_area.grid(row=0, column=1, sticky="nsew")

ctk.CTkLabel(main_area, text="YieldDoc", font=ctk.CTkFont(size=18, weight="bold"), text_color="white").pack(pady=10)

# Chat display area (scrollable frame for bubbles)
chat_scroll_frame = ctk.CTkScrollableFrame(main_area, fg_color="#414860")
chat_scroll_frame.pack(padx=10, pady=10, fill="both", expand=True)

# ==== INPUT AREA ====
input_frame = ctk.CTkFrame(main_area, fg_color="#414860")
input_frame.pack(fill="x", padx=10, pady=10)

user_input = ctk.CTkEntry(input_frame, placeholder_text="Type your message...", corner_radius=15)
user_input.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=5)

# Bind Enter key to send message
user_input.bind("<Return>", lambda event: send_message())

send_button = ctk.CTkButton(
    input_frame,
    text="Send",
    command=send_message,
    corner_radius=10,
    fg_color="#293046",
    hover_color="#1e2333"
)
send_button.pack(side="right", pady=5)

# Start the app
window.mainloop()