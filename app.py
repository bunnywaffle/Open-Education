import gradio as gr
from openai import OpenAI


PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;"> 
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">Ai Teacher</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Ask me anything...</p>
</div>
"""

#system prompt

system = """
Assume the role of a kind and non-judgmental AI tutor. Respond in a gentle and empathetic tone, always prioritizing the student's understanding and comfort. Break down complex concepts into clear, concise, and easy-to-understand language. Adapt your responses to the student's individual learning style, using a structured and organized approach to facilitate comprehension. Be patient, encouraging, and supportive in your language, fostering a safe and inclusive learning environment. Avoid using technical jargon or condescending language, instead focusing on empowering the student with confidence and clarity. Provide explanations that are relatable, accessible, and engaging, making the learning experience enjoyable and rewarding.
"""


# Set up the OpenAI API client with a local base URL and API key
client = OpenAI(base_url="http://localhost:5001/v1", api_key="none")


def chat(message, history):
    # Convert the conversation history into the format required by the OpenAI API
    history = [{"role": "system", "content": system }]
    for human, assistant in history:
        history.append({"role": "user", "content": human })  # Add the user's message
        history.append({"role": "teacher", "content": assistant})  # Add the assistant's response
    history.append({"role": "user", "content": message})  # Add the current user message

    # Use the OpenAI API to generate a response
    response = client.chat.completions.create(
        model='model',  # Use model
        messages=history,  # Pass in the conversation history
        temperature=0.8,  # Control the creativity of the response
        stream=True  # Stream the response in chunks
    )

    # Process the response in chunks and yield the partial message
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message

# Create a Gradio chat interface with the history function
gr.ChatInterface(
    chat,  # The predict function that generates responses
    chatbot=gr.Chatbot(height=500, placeholder=PLACEHOLDER ),  # Customize the chatbot appearance
    textbox=gr.Textbox(  # Customize the text input box
        placeholder="Ask Anything I am not a human; I don't judge.",
        container=False,
        scale=20,
    ),
    title="AI Tutor",  # Set the title of the chat interface
    theme="soft",  # Choose a theme for the interface
    examples=["Help me", "Are tomatoes vegetables?"],  # Provide some example inputs
    cache_examples=False,  # Don't cache the example inputs
    undo_btn="Delete Previous",  # Customize the undo button
    clear_btn="Clear",  # Customize the clear button
    submit_btn="Ask",  # Customize the submit button
).launch()  # Launch the chat interface
