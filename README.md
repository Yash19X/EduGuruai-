# Install dependencies (only first time)
!pip install gradio transformers accelerate sentencepiece

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose a small open-source model (free on Hugging Face)
# Options: "mistralai/Mistral-7B-Instruct-v0.2" or "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# Function for chatbot
def chat_with_ai(user_input, history):
    # Add system instruction: act like a teacher
    prompt = f"You are EduGuruAI, an AI tutor. Explain concepts clearly like a teacher.\n\nUser: {user_input}\nEduGuruAI:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.9
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply.split("EduGuruAI:")[-1].strip()

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📚 EduGuruAI – Your Personal AI Tutor")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask me any question... (e.g. Explain Newton's Laws)")
    clear = gr.Button("Clear Chat")

    def user_message(user_input, history):
        history = history + [(user_input, chat_with_ai(user_input, history))]
        return history, ""

    msg.submit(user_message, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()# EduGuruai-
