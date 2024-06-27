import os
from threading import Thread

from dotenv import load_dotenv
from flask import Flask, send_from_directory
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from werkzeug.utils import secure_filename

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit.components.v1 import html

# Load environment variables
from dotenv import load_dotenv()
load_dotenv()

# Set Streamlit configuration
st.set_page_config(
    page_title="AI-DOCTOR Agent",
    page_icon="🧑‍⚕️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

with st.sidebar:
    st.sidebar.image("doctor photo.png", caption="", use_column_width=True)
    selected=option_menu('Menu Options',
                         ['Home','Medical Report Summery','Dr. HeAlify Bot'],
                         icons=['person-circle','file-earmark-medical-fill'],
                         default_index=0)
    

#-------------------------------------------------------------------------------------------------------------------------------------------------------

if selected == 'Home':
    st.title("Welcome to HeAlify Doc-Agent 🧑‍⚕️")
    st.image("HeAlify main logo.png", use_column_width=True)
    st.write("""
    Our mission is to provide you with a seamless and advanced AI-powered healthcare assistant that can help you understand your medical reports, provide personalized health advice, and assist you with any medical inquiries you might have.

    **Features:**
    - **Medical Report Summarization:** Upload your medical reports and get concise summaries, key insights, and actionable advice.
    - **Dr. HeAlify Bot:** Chat with our AI-powered doctor for real-time assistance and medical advice.

    We are committed to improving your healthcare experience by leveraging the power of artificial intelligence.

    **How to Get Started:**
    - Navigate to the **Medical Report Summery** section to upload and summarize your medical reports.
    - Visit the **Dr. HeAlify Bot** section to chat with our AI doctor for any medical advice or questions.

    Thank you for choosing HeAlify Doctor Agent. Your health is our priority!

    **Contact Us:**
    If you have any questions or need assistance, feel free to contact me at [CLICK ME](kartavyamaster17@gma).
    """)
    st.header('', divider='rainbow')
    with st.container(border=True): 
        st.markdown('''
            Developed by KARTAVYA MASTER :8ball:
        ''')
        link = 'PORTFOLIO : [CLICK ME](https://mydawjbhdas.my.canva.site/aiwithkartavya)'
        st.markdown(link, unsafe_allow_html=True)



#-------------------------------------------------------------------------------------------------------------------------------------------------------

if selected=='Medical Report Summery':
    # Initialize title of the Streamlit app
    st.title("Medical Report Summarization 🧑‍⚕️")

    st.markdown("Here, you can upload your medical reports in PDF format and receive a detailed, concise summary of key findings, diagnoses, treatments, and recommendations. Our advanced AI analyzes your reports to provide you with clear insights and actionable advice. 🤖📋")


    # Load Groq API key from environment variable
    groq_api_key = os.getenv("GROQ_API_KEY")

    # Initialize the ChatGroq language model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

    # Define the prompt template for the language model
    prompt_template = """
    You are an advanced AI agent designed to process and summarize medical reports, as well as provide useful insights,Suggestions,Advice, Care Instructions based on the information provided.
    Your task involves:

    Tasks:
    - Concisely summarize the key findings, diagnoses, treatments, and recommendations outlined in the medical report.
    - Highlight any significant medical history, lab results, imaging findings, and physical examination details.

    Insight Extraction:
    - Answer specific questions related to the patient's condition, prognosis, and treatment options based on the report.
    - Provide explanations for medical terms and conditions mentioned in the report.
    - Suggest possible next steps or further tests if the information in the report indicates a need for them.
    - Identify any potential red flags or urgent issues that require immediate attention.

    Medical Report Summary:
    - Provide a concise summary of the key findings, diagnoses, treatments, and recommendations outlined in the medical report.
    - Highlight significant medical history, lab results, imaging findings, and physical examination details.

    Patient Information and Insights:
    - Answer specific questions related to the patient's condition, prognosis, and treatment options based on the report.
    - Provide explanations for medical terms and conditions mentioned in the report.

    Suggestions and Recommendations:
    - Suggest possible next steps or further tests if the information in the report indicates a need for them.
    - Recommend appropriate medications and dosages based on the patient's condition and medical history.

    Advice and Care Instructions:
    - Offer health care advice and precautions tailored to the patient's condition.
    - Provide detailed instructions on how the patient can manage their condition at home.
    - Highlight any potential red flags or urgent issues that require immediate attention.

    Format your response in a structured and readable manner. Use bullet points and sections where necessary to enhance clarity.use Incorporate emojis to make the advice more engaging and visually appealing.

    <context>
    {context}
    </context>

    respond the above questions
    Summarize the medical report provided above.
    Suggest any additional tests or consultations if necessary.
    Identify and elaborate on any urgent issues that need immediate attention.
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Set up the file uploader widget in Streamlit
    uploaded_file = st.file_uploader("Upload your medical report PDF file", type="pdf")

    # Define the upload folder for saving the uploaded files
    UPLOAD_FOLDER = 'uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Function to run Flask server
    def run_flask():
        app = Flask(__name__)

        @app.route('/uploads/<filename>')
        def uploaded_file(filename):
            return send_from_directory(UPLOAD_FOLDER, filename)

        app.run(port=8000)

    # Start Flask server in a separate thread if not already started
    if 'flask_thread' not in st.session_state:
        flask_thread = Thread(target=run_flask)
        flask_thread.start()
        st.session_state['flask_thread'] = flask_thread

    # Initialize session state variables if they don't exist
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
    if 'text_splitter' not in st.session_state:
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Process the uploaded file
    if uploaded_file is not None:
        # Secure the file name
        filename = secure_filename(uploaded_file.name)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Generate the file URL
        file_url = f"http://localhost:8000/uploads/{filename}"
        
        # Update session state with the file path
        st.session_state.file_path = file_path
        
        # Load documents and create vectors
        st.session_state.loader = PyPDFLoader(st.session_state.file_path)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        # Define the secondary prompt for insights and advice
        secondary_prompt = """
        What are some insights you can carry out from the PDF? Give me precautions and health care advice. Use emojis to look attractive.
        Write in a structured and readable format.
        """
        
        # Create document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Prepare context with input
        context = {'input': prompt_template, 'context': st.session_state.docs}  # Update context with required input key

        # Get the response from the retrieval chain
        response = retrieval_chain.invoke(context)
        
        # Display the response in Streamlit
        with st.container():
            st.write(response['answer'])

        # Provide a download button for the medical summary
        st.download_button(
            label="Download Your Medical Summary",
            data=response['answer'],
            file_name="Medical_Report_Summary.txt",
            mime="text/plain",
        )
    else:
        st.write("Please upload a medical report PDF file to get started.")

#-------------------------------------------------------------------------------------------------------------------------------------------------------

if selected=='Dr. HeAlify Bot':
    import streamlit as st
    import os
    from dotenv import load_dotenv
    from PIL import Image
    import google.generativeai as genai

    # Load environment variables from .env file
    load_dotenv()

    # Define the paths to the images
    
    chatbot_url = "chatbot photo.png"

    st.title("Dr. HeAlify Bot 🤖🩺")
    st.image("chatbot photo.png", use_column_width=True)
    st.write("""
    Welcome to the **Dr. HeAlify Bot** section! Here, you can interact with our advanced AI-powered chatbot designed to provide you with instant medical assistance and advice.

    **Meet Dr. HeAlify Bot:**
    Dr. HeAlify Bot is a virtual healthcare assistant that leverages the latest AI technology to help answer your medical queries, provide health tips, and guide you on your healthcare journey. Whether you need information about symptoms, medications, or general health advice, Dr. HeAlify Bot is here to assist you 24/7.
    """)

    st.info("Start chatting with Dr. HeAlify Bot now and get the medical assistance you need! 💬🩺")

    # Define user and chatbot roles
    USER = "user"
    ASSISTANT = "assistant"

    # Initialize chat history in session state (empty list)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Configure the Gemini API using the API key from environment variables
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Initialize the generative model
    model = genai.GenerativeModel("gemini-1.5-pro")

    # Define the input prompt for the generative model
    input_prompt = """
    You are a world-class doctor and your name is Dr. HeAlify Bot. Your goal is to assist the patient by understanding their medical problem, asking relevant questions one by one, and providing useful suggestions and advice.

    When a patient presents a medical problem, you should:
    1. Ask the patient detailed questions about their symptoms, duration.
    2. Inquire about any relevant medical history, and known allergies.
    3. Provide possible diagnoses based on the patient's responses.
    4. Suggest appropriate treatments, medications, or home remedies.
    5. Recommend any necessary further tests or consultations with specialists.
    6. Offer general health advice and preventive measures.

    Ensure your responses are clear, empathetic, and professional. Use simple language and provide explanations where necessary. Format your response with sections and bullet points to enhance readability.

    Now, respond to the patient's query.
    """

    # Function to get response from Gemini model
    def get_gemini_response(input_prompt, user_input):
        response = model.generate_content([input_prompt, user_input])
        return response.text

    # Get user input
    user_input = st.chat_input("Type your message here...")

    # Add user input to chat history
    if user_input:
        st.session_state["chat_history"].append({"role": USER, "content": user_input})

    # Display chat history
    for message in st.session_state["chat_history"]:
        st.chat_message(message["role"]).write(message["content"])

    # Process user input and respond with Gemini
    if user_input:
        gemini_response = get_gemini_response(input_prompt, user_input)
        st.session_state["chat_history"].append({"role": ASSISTANT, "content": gemini_response})
        st.chat_message(ASSISTANT).write(gemini_response)
