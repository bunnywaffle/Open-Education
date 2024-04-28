import gradio as gr
from openai import OpenAI

# Set up the OpenAI API client with a local base URL and API key
client = OpenAI(base_url="http://localhost:5001/v1", api_key="none")

# Define the prompt that will be used to guide the AI tutor's responses
Prompt = """
Act as a compassionate and knowledgeable AI tutor, prioritizing the student's comprehension and emotional comfort. When responding, adhere to the following guidelines:

Break down complex concepts: Divide intricate ideas into manageable, easy-to-understand components, using relatable analogies and examples to facilitate understanding.
Encourage and motivate: Offer supportive and uplifting comments to foster a sense of confidence and curiosity, urging the student to explore and learn.
Experimentation and testing: Design interactive exercises and thought-provoking questions that encourage the student to apply their knowledge, experiment with concepts, and receive feedback on their progress.
Personalized learning: Adapt your teaching approach to accommodate the student's individual learning style, pace, and preferences, ensuring a tailored and effective learning experience.
Clear and structured responses: Organize your answers in a logical, step-by-step manner, using concise language and avoiding ambiguity, to promote clarity and facilitate understanding.
Supportive tone: Maintain a warm, empathetic, and non-judgmental tone, providing reassurance and guidance throughout the learning process.
"""

#placeholder

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;"> 
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">Ai Teacher</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Ask me anything...</p>
</div>
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
    chatbot=gr.Chatbot(height=500, placeholder=PLACEHOLDER),  # Customize the chatbot appearance
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
              "How do you evaluate the credibility of a source in research?",
              "Can you explain the process of speciation, and how does it occur?",
              "What is the current understanding of the origin of life on Earth?"
              ],  # Provide some example inputs
    cache_examples=False,  # Don't cache the example inputs
    undo_btn="Delete Previous",  # Customize the undo button
    clear_btn="Clear",  # Customize the clear button
    submit_btn="Ask",  # Customize the submit button
).launch()  # Launch the chat interface
