# FarmerAI
A LLM designed specifically for Farmer's to ask it questions regarding plant information, harvest yield and other such questions.

## Model
deepseek-r1:1.5b

- A small model used for local deployment of Deepseek's R1 model.

# Functional Tasks
### Task 1: Create GUI for our application using Tkinter or other Python-based GUI library. 
- Should include text box to type in, an enter button to submit your text to the AI and a textbox to show the AI's response
- Style to look like a messeging app ( chatgpt or deepseek)
- Add method to track "history" of chats

### Task 2: Enable the user to use speech-to-text to talk to the AI instead of typing in the text box.
- Should include a start talking and stop talking button (or use one that toggles between those).
- Optional: Enable the user to use a keyword to capture text and then another keyword to stop voice capture.

### Task 3: Enable the AI model to speak to the user when it responds.
- Should include a toggle button to allow/disallow the AI to speak based on user preference.

# Studying / Data Tasks
### Task 1: Study the usage of irrigation based on drought status / climate of hte specific region of the farmer and how this can be implemented into our AI agent.

### Task 2: Source dataset(s) for environmental data and farming practices for differing climates: drought vs. no drought areas.

### Task 3: Source dataset(s) for agricultural information regardig what crops should be planted after what other crops (crop rotation data).

### Task 4: Source dataset(s) for agricultural information regarding what plants grow best in what soil types, climates, regions, etc.


# How to download the model we use after installing Ollama to your computer

Run the following command to pull the 1GB model from Ollama: <br>
`ollama run deepseek-r1:1.5b`

# How to stop and start up the model

## 1. Stopping the Model

To stop a model that's currently running and free up your computer's CPU resources, you can use the ollama stop command.<br>

`ollama stop <model_name>`<br>

Replace `<model_name>` with the name of the model that's running. If you're unsure of the model name, you can check the running models using:<br>

`ollama ps`<br>

This will list all the running models, and you can stop the one you want by specifying its name.

## 2. Starting the Model

To restart or start a model again after stopping it, use the ollama serve command:

`ollama serve <model_name>`

This will start the model and make it available for you to use again.
