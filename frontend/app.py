import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import customtkinter as ctk
import time
from peft import PeftModel
import datetime

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3")

# Load the fine-tuned model with the adapter
model = PeftModel.from_pretrained(base_model, "czalpha/fine_tuned_model")

# Use the tokenizer from the base model (since it's the same as the fine-tuned one)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

# Set appearance mode and theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

def get_timestamp():
    """Returns a timestamp string."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def send_message():
    user_msg = user_input.get()
    if user_msg.strip() == "":
        return

    timestamp = get_timestamp()
    chat_display.configure(state="normal")
    chat_display.insert("end", f"You ({timestamp}): {user_msg}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(user_msg, return_tensors="pt").to(device)

    output_tokens = model.generate(**inputs, max_length=100)
    full_response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Trim the response to exclude the input prompt
    ai_response = full_response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]

    timestamp = get_timestamp()
    chat_display.configure(state="normal")
    chat_display.insert("end", f"AI ({timestamp}): {ai_response}\n")
    chat_display.configure(state="disabled")
    chat_display.see("end")
    user_input.delete(0, "end")

# Main window setup
window = ctk.CTk()
window.title("AI Messaging App")
window.geometry("800x500")
window.configure(fg_color="#414860")

# Layout configuration
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=3)
window.grid_rowconfigure(0, weight=1)

side_panel = ctk.CTkFrame(window, width=250, fg_color="#2e344f", corner_radius=0)
side_panel.grid(row=0, column=0, sticky="nsew")
side_panel.grid_propagate(False)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=0)

ctk.CTkLabel(side_panel, text="History", text_color="white", font=ctk.CTkFont(size=16), anchor="w").pack(fill="x", pady=10, padx=60)

main_area = ctk.CTkFrame(window, fg_color="#414860")
main_area.grid(row=0, column=1, sticky="nsew")

ctk.CTkLabel(main_area, text="YieldDoc", font=ctk.CTkFont(size=18, weight="bold"), text_color="white").pack(pady=10)

chat_display = ctk.CTkTextbox(main_area, height=300, fg_color="white", text_color="black", corner_radius=10)
chat_display.pack(padx=10, pady=10, fill="both", expand=True)
chat_display.configure(state="disabled")

input_frame = ctk.CTkFrame(main_area, fg_color="#414860")
input_frame.pack(fill="x", padx=10, pady=10)

user_input = ctk.CTkEntry(input_frame, placeholder_text="Type your message...", corner_radius=15)
user_input.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=5)

send_button = ctk.CTkButton(input_frame, text="Send", command=send_message, corner_radius=10, fg_color="#293046", hover_color="#1e2333")
send_button.pack(side="right", pady=5)

window.mainloop()