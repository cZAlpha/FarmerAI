import customtkinter as ctk

# App settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Data to store chats
chat_sessions = []
current_chat_index = -1

# Function to start a new chat
def new_chat():
    global current_chat_index
    current_chat_index = len(chat_sessions)
    chat_sessions.append([])  # Add empty session
    update_history_buttons()
    clear_chat_display()

def clear_chat_display():
    for widget in chat_scroll_frame.winfo_children():
        widget.destroy()

# Function to send message
def send_message():
    global current_chat_index
    user_msg = user_input.get()
    if user_msg.strip() == "":
        return

    ai_response = "I'm just a mock AI. How can I help you?"

    if current_chat_index == -1:
        new_chat()

    # Save messages to session
    chat_sessions[current_chat_index].append(("You", user_msg))
    chat_sessions[current_chat_index].append(("AI", ai_response))

    # Display messages as bubbles
    add_message_bubble("You", user_msg)
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
        pady=6
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
