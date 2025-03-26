import customtkinter as ctk

# Set appearance mode and theme
ctk.set_appearance_mode("dark")  # Options: "light", "dark", "system"
ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

# Function to simulate AI chat
def send_message():
    user_msg = user_input.get()
    if user_msg.strip() == "":
        return

    chat_display.configure(state="normal")
    chat_display.insert("end", f"You: {user_msg}\n")

    ai_response = "AI: I'm just a mock AI. How can I help you?\n"
    chat_display.insert("end", ai_response)

    chat_display.configure(state="disabled")
    chat_display.see("end")
    user_input.delete(0, "end")


# Main window setup
window = ctk.CTk()
window.title("AI Messaging App")
window.geometry("800x500")
window.configure(fg_color="#414860")

# Layout configuration
window.grid_columnconfigure(0, weight=1)  # side panel
window.grid_columnconfigure(1, weight=3)  # main area
window.grid_rowconfigure(0, weight=1)

side_panel = ctk.CTkFrame(window, width=250, fg_color="#2e344f", corner_radius=0)
side_panel.grid(row=0, column=0, sticky="nsew")  # changed to nsew to fully expand
side_panel.grid_propagate(False)

# Ensure it fills all space
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=0)

# Optional: add fill to label if you want it to stretch
ctk.CTkLabel(
    side_panel,
    text="History",
    text_color="white",
    font=ctk.CTkFont(size=16),
    anchor="w"
).pack(fill="x", pady=10, padx=60)

# Main chat area
main_area = ctk.CTkFrame(window, fg_color="#414860")
main_area.grid(row=0, column=1, sticky="nsew")

# Title label
ctk.CTkLabel(main_area, text="YieldDoc", font=ctk.CTkFont(size=18, weight="bold"), text_color="white").pack(pady=10)

# Chat display area
chat_display = ctk.CTkTextbox(
    main_area,
    height=300,
    fg_color="white",
    text_color="black",
    corner_radius=10
)
chat_display.pack(padx=10, pady=10, fill="both", expand=True)
chat_display.configure(state="disabled")

# Input area
input_frame = ctk.CTkFrame(main_area, fg_color="#414860")
input_frame.pack(fill="x", padx=10, pady=10)

user_input = ctk.CTkEntry(
    input_frame,
    placeholder_text="Type your message...",
    corner_radius=15
)
user_input.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=5)

send_button = ctk.CTkButton(
    input_frame,
    text="Send",
    command=send_message,
    corner_radius=10,
    fg_color="#293046",
    hover_color="#1e2333"
)
send_button.pack(side="right", pady=5)

# Run the app
window.mainloop()
