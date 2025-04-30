# FarmerAI

## Overview
FarmerAI is a Python-based chatbot application designed to answer agriculture-related questions. The application utilizes various language models to provide helpful responses to farming queries. Users can choose between different AI models before starting a conversation with the bot.

## Features
- Interactive chat interface using CustomTkinter
- Multiple AI model options
- Real-time responses to agriculture questions
- GPU acceleration when available

## Requirements
- Python 3.8+
- Torch
- Transformers
- Accelerate
- CustomTkinter
- Internet connection (for initial model download)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/farmerai.git
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows:
   ```bash
   venv\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. Install the required packages:
```bash
pip install torch transformers accelerate customtkinter
```

## Usage

1. Run the application:
```bash
python farmerai.py
```

2. Select a model when prompted:
   - Option 1: Our custom Mistral-based model (Currently marked as not working)
   - Option 2: bagasbgs2516's Llama2-based agriculture model (Currently marked as not working)
   - Option 3: AgriQBot by mrSoul7766 (Working)

3. Once the model is loaded, you can start asking agriculture-related questions through the chat interface.

## Available Models

### Our Model (Option 1)
- Base model: mistralai/Mistral-7B-v0.3
- Fine-tuned model: czalpha/fine_tuned_model
- Status: Currently not working

### bagasbgs2516's Model (Option 2)
- Base model: meta-llama/Llama-2-7b-hf
- Fine-tuned model: bagasbgs2516/llama2-agriculture-lora
- Status: Currently not working

### AgriQBot (Option 3)
- Creator: mrSoul7766
- Model link: https://huggingface.co/mrSoul7766/AgriQBot
- Dataset: https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only
- Status: Working

## Technical Details

The application uses:
- 8-bit quantization for efficient memory usage
- CPU offloading when necessary
- GPU acceleration when available
- Automatic model caching for faster subsequent loads

## Future Improvements
- Fix non-working models
- Add more agriculture-specific models
- Implement conversation history saving
- Enhance the user interface
- Add offline mode support

## License
[Add your license information here]

## Acknowledgments
- mrSoul7766 for creating AgriQBot
- bagasbgs2516 for their agriculture fine-tuned Llama2 model
- The Hugging Face team for their Transformers library

---

For issues, feature requests, or contributions, please open an issue on the GitHub repository.
