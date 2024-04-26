import gradio as gr
from openai import OpenAI

# Set up the OpenAI API client
client = OpenAI(base_url="http://localhost:5001/v1", api_key="none")


def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "system", "content": "I am a educational tutor" })
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})
  
    response = client.chat.completions.create(model='gpt-3.5-turbo',
    messages= history_openai_format,
    temperature=1.0,
    stream=True)

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
              partial_message = partial_message + chunk.choices[0].delta.content
              yield partial_message

gr.ChatInterface(predict,     
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title="AI Tutor",
    theme="soft",
    examples=["Help me", "Are tomatoes vegetables?"],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear", submit_btn="Ask",).launch()
