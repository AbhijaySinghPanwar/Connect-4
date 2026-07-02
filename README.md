---
title: Connect
emoji: 🔵
colorFrom: blue
colorTo: blue
sdk: gradio
app_file: app.py
sdk_version: 5.15.0
pinned: false
python_version: 3.12
short_description: Arena for playing Four-in-a-row between LLMs
---

# Four-in-a-row Arena

### A battleground for pitting LLMs against each other in the classic board game

It has been great fun making this Arena and watching LLMs duke it out!

Quick links:
- The [GitHub repo](https://github.com/AbhijaySinghPanwar/Connect-4) for the code

## Installing the code

1. Clone the repo with `git clone https://github.com/AbhijaySinghPanwar/Connect-4.git`
2. Change to the project directory with `cd Connect-4`
3. Create a python virtualenv with `python -m venv venv`
4. Activate your environment with either `venv\Scripts\activate` on Windows, or `source venv/bin/activate` on Mac/Linux
5. Then run `pip install -r requirements.txt` to install the packages

To launch the app locally, run `python app.py`

## Setting up your API keys

Please create a file with the exact name `.env` in the project root directory.

You would typically use Notepad (Windows) or nano (Mac) for this.

Your .env file should contain the following; add whichever keys you would like to use.

```
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk...
GROQ_API_KEY=...
```

## Optional - using Ollama

You can run Ollama locally, and the Arena will connect to run local models.  
1. Download and install Ollama from https://ollama.com noting that on a PC you might need to have administrator permissions for the install to work properly
2. On a PC, start a Command prompt / Powershell (Press Win + R, type `cmd`, and press Enter). On a Mac, start a Terminal (Applications > Utilities > Terminal).
3. Run `ollama run llama3.2` or for smaller machines try `ollama run llama3.2:1b`
4. If this doesn't work, you may need to run `ollama serve` in another Powershell (Windows) or Terminal (Mac), and try step 3 again
