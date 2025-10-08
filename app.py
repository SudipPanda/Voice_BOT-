
import gradio as gr
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import openai
import tempfile
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate # Import PromptTemplate
import os

# ---- OpenAI API setup ----
# If you want a free alternative, you can replace openai.ChatCompletion with a dummy function
# from google.colab import userdata
# API_key = userdata.get('OPENAI_API_KEY')

API_key = os.getenv("OPENAI_API_KEY")
if not API_key:
    raise ValueError("Please set your OPENAI_API_KEY as an environment variable.")

os.environ["OPENAI_API_KEY"] = API_key

from openai import OpenAI

with open("personal_info.txt", "r") as f:
    docs = f.read()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = text_splitter.split_text(docs)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_texts(chunks, embeddings)
retriever = vectordb.as_retriever()

# Define a custom prompt template
template = (
    "You are a friendly, casual, and humorous 20-year-old boy named Sudip Panda."
    "Speak like a young adult with lively and relatable language."
   "When answering, first check if the retrieved documents have relevant info."
   "If they do, use it."
   "If the question is unrelated to the documents, answer naturally based on general knowledge, not just AI/ML."
   "Keep answers informal and energetic."
    "DOn't use any emoji in response"
    "\n\n{context}"
    "\n\nQuestion: {question}"
    "\nHelpful Answer:"
)
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo" , temperature=0.7),
    chain_type="refine",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"refine_prompt": QA_CHAIN_PROMPT} # Pass the prompt template here
)


client = OpenAI(api_key=API_key)

def ask_chatgpt(prompt):
    responses = qa_chain(prompt)
    print(responses)
    return responses

#OK TO GO HERE

def speech_to_text(audio):
    """
    Converts audio input to text using SpeechRecognition.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            #print("the value of the text is ",text)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Speech Recognition service error."

#ok to go
def text_to_speech(text):
    """
    Convert text to audio and return a filepath for Gradio Audio output.
    """
    tts = gTTS(text=text, lang="en")
    # Save to temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name  # Return filepath for Gradio

#ok to go

def chatbot(audio_path):
    # audio_path is now a valid filename
    print(type(audio_path), audio_path)
    user_text = speech_to_text(audio_path)

    print("Recognized Text:", user_text)  # debug

    if "could not understand" in user_text:
        return user_text, None

    response = ask_chatgpt(user_text)
    response_text = response['result']
    response_audio = text_to_speech(response_text)
    return response_text, response_audio


# ---- Gradio interface ----
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Audio(sources="microphone", type="filepath", format="wav"),
    outputs=[gr.Textbox(label="ChatGPT Response"), gr.Audio(label="Voice Response" , autoplay=True)],
    title="Voice ChatGPT Bot",
    description="Ask me anything! Speak into your mic, and I will respond in text and voice."
)

iface.launch(debug=  True, share=True)
