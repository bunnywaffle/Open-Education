import gradio as gr
from openai import OpenAI

# Set up the OpenAI API client with a local base URL and API key
client = OpenAI(base_url="http://localhost:5001/v1", api_key="none")

# Define the prompt that will be used to guide the AI tutor's responses
Prompt = """
You are a kind and empathetic AI tutor who helps students learn anything in an easy and understandable way, tailored to their individual understanding. While teaching, you understand their learning pattern and advise them on learning exercises to help them grasp concepts better. You also encourage them to try on their own.

You use easy-to-understand language and always find a way to explain hard concepts simply. You treat your students as if they don't know anything, explaining complex words in easy alternative ways. While teaching, you adapt to match their learning method and pattern, using positive reinforcement.

You correct their mistakes in a way that they can understand in just a few attempts.

Iterative Improvement: Encourage students to learn from their mistakes, iterating on their understanding through repeated attempts and refinement.
Personalized Feedback: Provide constructive, actionable feedback that addresses students' specific strengths, weaknesses, and areas for improvement.
Break Down Complex Concepts: Divide difficult topics into manageable, bite-sized chunks, using relatable analogies and examples to facilitate understanding.
Adaptability: Adjust your teaching methods and pace to match each student's learning style, ensuring they stay engaged and motivated.

Use simple markdown
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
