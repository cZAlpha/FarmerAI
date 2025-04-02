print("")
print("[+] Initializing libraries used for this program...")
from transformers import AutoModelForCausalLM, AutoTokenizer
print("     [+] transformers has been imported.")
import customtkinter as ctk
print("     [+] customtkinter has been imported.")
from peft import PeftModel
print("     [+] peft has been imported.")
import datetime
print("     [+] datetime has been imported.")
import torch
print("     [+] datetime has been imported.")
import time
print("     [+] time has been imported.")
print("[+] Initialization has finished.")
print("")

# Handles picking what model to use for the application
valid_input = False
while not valid_input:
    print("Choose which model you would like to use:")
    print("1. Our Model")
    print("2. bagasbgs2516's Model")
    model_choice = input("> ")
    
    # Handle choices
    if model_choice == "1":
        our_model = True # Whether or not our model will be used
        valid_input = True # Exit loop
    elif model_choice == "2":
        our_model = False # Whether or not our model will be used
        valid_input = True # Exit loop
    else:
        print("INVALID INPUT. You must enter '1' or '2'.")
        time.sleep(3)
        for _ in range(20):
            print("")


if (our_model):
    print("")
    print("You have chosen to use our model.")
    print("")
    print("~ Information ~")
    print(" - Base Model: mistralai/Mistral-7B-v0.3")
    print(" - Our Model:  czalpha/fine_tuned_model")
    print(" - Tokenizer:  mistralai/Mistral-7B-v0.3")
    print("")
    # OUR MODEL
        # Load base model
    base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3")
        # Load the fine-tuned model with the adapter
    our_model = PeftModel.from_pretrained(base_model, "czalpha/fine_tuned_model")
        # Use the tokenizer from the base model (since it's the same as the fine-tuned one)
    our_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
else:
    print("")
    print("You have chosen to use bagasbgs2516's model.")
    print("")
    print("~ Information ~")
    print(" - Base Model: meta-llama/Llama-2-7b-hf")
    print(" - Our Model:  bagasbgs2516/llama2-agriculture-lora")
    print(" - Tokenizer:  bagasbgs2516/llama2-agriculture-lora")
    print("")
    # bagasbgs2516's MODEL
        # Load base model
    meta_base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        # Load the fine-tuned model with the adapter
    bagasbgs2516_model = PeftModel.from_pretrained(meta_base_model, "bagasbgs2516/llama2-agriculture-lora")
        # Load the tokenizer from the *local* directory, using the files he provided.
    bagasbgs2516_tokenizer = AutoTokenizer.from_pretrained("bagasbgs2516/llama2-agriculture-lora") 


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
    
    # Add your message to the window before ANY AI stuff
    chat_sessions[current_chat_index].append(("You", user_msg))
    add_message_bubble("You", user_msg)
    
    # Depending on the model selected by the user, calculate the AI's response
    if (our_model): # If we're using our model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = our_tokenizer(user_msg, return_tensors="pt").to(device)
            # Get the output tokens from the AI
        output_tokens = our_model.generate(**inputs, max_length=100) # CHANGE THIS BASED ON MODEL USAGE
            # Turn that into like english or something
        full_response = our_tokenizer.decode(output_tokens[0], skip_special_tokens=True) # CHANGE THIS BASED ON DIFFERING MODEL'S TOKENIZER
            # Trim the response to exclude the input prompt
        ai_response = full_response[len(our_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):] # CHANGE THIS BASED ON DIFFERING MODEL'S TOKENIZER
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = bagasbgs2516_tokenizer(user_msg, return_tensors="pt").to(device)
            # Get the output tokens from the AI
        output_tokens = bagasbgs2516_model.generate(**inputs, max_length=100) # CHANGE THIS BASED ON MODEL USAGE
            # Turn that into like english or something
        full_response = bagasbgs2516_tokenizer.decode(output_tokens[0], skip_special_tokens=True) # CHANGE THIS BASED ON DIFFERING MODEL'S TOKENIZER
            # Trim the response to exclude the input prompt
        ai_response = full_response[len(bagasbgs2516_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):] # CHANGE THIS BASED ON DIFFERING MODEL'S TOKENIZER
    
    # Save AI's to session
    chat_sessions[current_chat_index].append(("AI", ai_response))
    # Display messages as bubbles
    add_message_bubble("AI", ai_response)
    
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