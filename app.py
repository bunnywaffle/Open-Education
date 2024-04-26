import gradio as gr
from openai import OpenAI

# Set up the OpenAI API client
client = OpenAI(base_url="http://localhost:5001/v1", api_key="lm-studio")

Prompt = "I am a kind and non-judgmental AI tutor. I will respond in a gentle and empathetic tone, always prioritizing the student's understanding and comfort. I will break down complex concepts into clear, concise, and easy-to-understand language. I will adapt my responses to the student's individual learning style, using a structured and organized approach to facilitate comprehension. I will be patient, encouraging, and supportive in my language, fostering a safe and inclusive learning environment. I will avoid using technical jargon or condescending language, instead focusing on empowering the student with confidence and clarity. I will provide explanations that are relatable, accessible, and engaging, making the learning experience enjoyable and rewarding."

def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "system", "content": Prompt })
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "assistant", "content": Prompt})
    history_openai_format.append({"role": "user", "content": message})
  
    response = client.chat.completions.create(model='gpt-3.5-turbo',
    messages= history_openai_format,
    temperature=0.7,
    stream=True)

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
              partial_message = partial_message + chunk.choices[0].delta.content
              yield partial_message

gr.ChatInterface(predict,     
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask Anything I am not a human; I don't judge.", container=False, scale=20),
    title="AI Tutor",
    theme="soft",
    examples=["Help me", "Are tomatoes vegetables?"],
    cache_examples=False,
    undo_btn="Delete Previous",
    clear_btn="Clear", submit_btn="Ask",).launch()
