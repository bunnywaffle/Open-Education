import gradio as gr
from openai import OpenAI

# Set up the OpenAI API client with a local base URL and API key
client = OpenAI(base_url="http://localhost:5001/v1", api_key="none")

# Define the prompt that will be used to guide the AI tutor's responses
Prompt = """
You are a kind and non-judgmental AI tutor. you will respond in a gentle and empathetic tone, 
always prioritizing the student's understanding and comfort. you will break down complex concepts 
into clear, concise, and easy-to-understand language. you will adapt your responses to the student's 
individual learning style, using a structured and organized approach to facilitate comprehension. 
you will be patient, encouraging, and supportive in my language, fostering a safe and inclusive 
learning environment. you will avoid using technical jargon or condescending language, instead 
focusing on empowering the student with confidence and clarity. You will provide explanations that 
are relatable, accessible, and engaging, making the learning experience enjoyable and rewarding.
"""

def predict(message, history):
    # Convert the conversation history into the format required by the OpenAI API
    chat_history = [{"role": "system", "content": Prompt }]
    for human, assistant in history:
        chat_history.append({"role": "user", "content": human })  # Add the user's message
        chat_history.append({"role": "assistant", "content": assistant})  # Add the assistant's response
    chat_history.append({"role": "user", "content": message})  # Add the current user message

    # Use the OpenAI API to generate a response
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',  # Use the GPT-3.5 Turbo model
        messages=chat_history,  # Pass in the conversation history
        temperature=0.8,  # Control the creativity of the response
        stream=True  # Stream the response in chunks
    )

    # Process the response in chunks and yield the partial message
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message

# Create a Gradio chat interface with the predict function
gr.ChatInterface(
    predict,  # The predict function that generates responses
    chatbot=gr.Chatbot(height=500),  # Customize the chatbot appearance
    textbox=gr.Textbox(  # Customize the text input box
        placeholder="Ask Anything I am not a human; I don't judge.",
        container=False,
        scale=20
    ),
    title="AI Tutor",  # Set the title of the chat interface
    theme="soft",  # Choose a theme for the interface
    examples=["How does the human brain process and store memories?", 
              "Are tomatoes vegetables?",
              "Can you explain the concept of infinity in calculus?",
              "What's the difference between a hypothesis and a theory in science?",
              "How do you evaluate the credibility of a source in research?"
              ],  # Provide some example inputs
    cache_examples=False,  # Don't cache the example inputs
    undo_btn="Delete Previous",  # Customize the undo button
    clear_btn="Clear",  # Customize the clear button
    submit_btn="Ask",  # Customize the submit button
).launch()  # Launch the chat interface
