import os
from time import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import numpy as np
import warnings
import transformers
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
import gc
from langgraph.graph import START, StateGraph
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
import re
import time
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, NotRequired, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from pptx import Presentation
from PyPDF2 import PdfReader
from docx import Document
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
import asyncio

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")

model = None
embedding_model = None
gc.collect()
torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('pypdf').setLevel(logging.ERROR)
logger.info('Logging Initialised')

# FastAPI App
app = FastAPI()
app.mount("/public", StaticFiles(directory="public"), name="public")
# Thread Pool for Async Execution
executor = ThreadPoolExecutor(max_workers=12)  #


class Embeddings:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', device='cuda'):
        self.model_name = model_name
        self.device = device
        self.model_kwargs = {"device": "cuda"}
        self.embedding_model = self._HF_embedding()
        self.chunks_overlap = None
        self.chunk_size = None
        self.retriever = None
        self.retrieved_docs = None

    def _HF_embedding(self):
        return HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs=self.model_kwargs)

    def load_documents(self, directory_path, force_reload=False, chunk_size=1000, chunk_overlap=20):
        persist_directory = "./chroma_db3"
        if force_reload:
            try:
                logger.info(f'Loading documents from {directory_path}')
                self.chunk_size = chunk_size
                self.chunks_overlap = chunk_overlap
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                all_documents = []
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        logger.info(f'Loading document: {file}')
                        file_path = str(os.path.join(root, file))

                        if file.endswith('.pdf'):
                            loaded_file = PyPDFLoader(file_path).load()
                        elif file.endswith('.docx'):
                            loaded_file = UnstructuredWordDocumentLoader(file_path).load()
                        elif file.endswith('.pptx'):
                            loaded_file = UnstructuredPowerPointLoader(file_path).load()
                        else:
                            continue

                        all_documents.extend(text_splitter.split_documents(loaded_file))

                vector_store = Chroma.from_documents(documents=all_documents, embedding=self.embedding_model,
                                                     persist_directory=persist_directory)
                vector_store.persist()
                self.retriever = vector_store.as_retriever()
                return self.retriever
            except Exception as e:
                logger.error(f'Failed to load documents: {e}')
                raise
        else:
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=self.embedding_model)
            self.retriever = vector_store.as_retriever()

    def get_documents(self, question, top_k):
        if not self.retriever:
            raise ValueError("Documents must be loaded before they can be searched")
        # get relevant docs
        self.retrieved_docs = self.retriever.invoke(question) or []  # Get initial retrieved docs
        # Re-rank and select top K
        ranked_docs = self.rerank_documents(question, top_k=top_k) if self.retrieved_docs else []
        formatted_docs = format_documents(ranked_docs) if ranked_docs else []
        return formatted_docs

    def rerank_documents(self, query, top_k=2):
        """Ranks documents based on cosine similarity to the query embedding."""
        if not self.embedding_model or hasattr(self.embedding_model, 'embed_query') is False:
            raise ValueError("Tokenizer must be loaded before configuring model kwargs")

        query_embedding = self.embedding_model.embed_query(query)  # Get query vector
        doc_embeddings = np.array(
            [self.embedding_model.embed_query(doc.page_content) for doc in self.retrieved_docs])  # Get doc vectors
        # Compute cosine similarity
        similarities = np.dot(doc_embeddings, query_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        # Rank by similarity and select top_k
        ranked_indices = np.argsort(similarities)[::-1]  # Sort in descending order
        top_docs = [self.retrieved_docs[i] for i in ranked_indices[:top_k]]
        return top_docs


class LLM:
    SUPPORTED_PRECISIONS = {'4bit', '8bit'}

    def __init__(self, HF_model_id="ibm-granite/granite-3.2-8b-instruct",
                 # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", meta-llama/Llama-3.1-8B
                 HF_token, device='cuda', load_precision='4bit'):
        self.HF_model_id = HF_model_id
        self.HF_token = HF_token
        self.device = device

        if load_precision not in self.SUPPORTED_PRECISIONS:
            raise ValueError(f"Invalid load_precision: {load_precision}. Must be '4bit' or '8bit'")
        self.load_precision = load_precision
        self.compute_precision = torch.float16  # make as input?

        self.model_config = None
        self.quantization_config = None
        self.model = None
        self.tokenizer = None
        self.query_pipeline = None
        self.HF_pipeline = None

        self.initialise_model_and_pipeline()

    def load_model(self):
        time_1 = time.time()
        self.model_config = AutoConfig.from_pretrained(
            self.HF_model_id, trust_remote_code=True, token=self.HF_token
        )

        load_in_4bit = True if self.load_precision == '4bit' else False
        load_in_8bit = True if self.load_precision == '8bit' else False

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=self.compute_precision
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.HF_model_id,
            trust_remote_code=True,
            config=self.model_config,
            quantization_config=self.quantization_config,
            device_map='auto',
            token=self.HF_token
        )
        logger.info(f"Model Loaded in: {round(time.time() - time_1, 3)} sec.")
        return self.model

    def configure_model_kwargs(self):
        if not self.tokenizer:
            raise ValueError("Tokenizer must be loaded before configuring model kwargs")

        return {
            "do_sample": False,  # Enable sampling for creativity
            "temperature": 0.2,  # Add temperature for balanced randomness
            #"top_k": 10,  # Increase top_k for more token options
            #"top_p": 0.9,  # Add top_p for nucleus sampling
            "repetition_penalty": 1.1,  # Slightly lower to allow natural repetition
            # "return_full_text": False,   # Keep as False to get only the response
            "num_return_sequences": 1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 500,  # Set a reasonable limit for detailed responses
            "max_length": 1000
        }

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.HF_model_id, token=self.HF_token)
        return self.tokenizer

    def init_pipeline(self):
        self.query_pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=self.compute_precision,
            # device='cuda:0',
            return_full_text=False, max_length=10000
        )

        self.HF_pipeline = HuggingFacePipeline(pipeline=self.query_pipeline, model_kwargs=self.configure_model_kwargs())

        return self.HF_pipeline

    def initialise_model_and_pipeline(self):
        logger.info('Loading model and pipeline')
        self.model = None
        self.tokenizer = None
        self.query_pipeline = None
        self.HF_pipeline = None

        self.load_model()
        self.load_tokenizer()
        self.init_pipeline()
        logger.info('Load Complete')


def parse_response(response: str):
    logger.debug(f"Raw model response: {response}")
    match = re.search(r"### Thought Process:\n(.*?)\n\n### Final Answer:\n(.*)", response, re.DOTALL)
    if match:
        reasoning, answer = match.groups()
        return reasoning.strip(), answer.strip()
    # Fallback: Split on any occurrence of "Final Answer" (case-insensitive)
    parts = re.split(r"(?i)### Final Answer:?\s*\n", response, maxsplit=1)
    if len(parts) > 1:
        reasoning, answer = parts[0].strip(), parts[1].strip()
        if not reasoning.startswith("### Thought Process:"):
            reasoning = f"### Thought Process:\n{reasoning}"
        return reasoning, answer
    # Last resort: Treat all but last line as reasoning
    lines = response.strip().split("\n")
    if len(lines) > 1:
        return "### Thought Process:\n" + "\n".join(lines[:-1]).strip(), lines[-1].strip()
    return "### Thought Process:\nReasoning not provided by model.", response.strip()


def format_documents(docs):
    """Formats retrieved documents for inclusion in the prompt."""
    if not docs:
        return "No relevant documents found."
    formatted_docs = []
    for doc in docs:
        if hasattr(doc, "metadata") and hasattr(doc, "page_content"):
            title = doc.metadata.get("title", "Unknown")
            formatted_docs.append(f"Title: {title}\nContent: {doc.page_content}")
        else:
            formatted_docs.append(str(doc))  # Handle unexpected types
    return "\n\n".join(formatted_docs)


# Define the function that calls the model
class LLMModelManager:
    def __init__(self, dir_path, document_force_reload=False):
        logger.info('Initialising LLMModelManger')
        self.llm = LLM()
        logger.info('LLM initialised')
        self.embeddings = Embeddings()
        logger.info('Embeddings object created')
        self.embeddings.load_documents(directory_path=dir_path, force_reload=document_force_reload)
        self.prompt = self._create_prompt()
        self.runnable = self.prompt | self.llm.HF_pipeline

    def _create_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             "### STRICT Answer Format Requirement: You MUST format your response EXACTLY as follows, with no deviations:\n\n### Thought Process:\n[Your step-by-step reasoning here]\n\n### Final Answer:\n[Your concise answer here]\n\nDo NOT include additional text, repeat the question, or deviate from this structure."),
            ("system",
             "### Instructions: Answer only the question provided. Before answering, think step by step and explain your reasoning in the 'Thought Process' section. Provide a concise answer in the 'Final Answer' section. Example:\n\nQuestion: What is 2+2?\nResponse:\n### Thought Process:\nFirst, I identify the numbers: 2 and 2. Then, I add them together: 2 + 2 = 4.\n\n### Final Answer:\n4"),
            ("system", "### Language Instructions: Respond in {language}."),
            ("system", "### Retrieved and Ranked Documents relevant to the question:\n{documents}."),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "### User Question:\n{question}"),
        ])

    def call_model(self, state: BaseModel):
        # Extract relevant fields from the state
        documents = state.documents if state.documents else ""
        language = state.language if state.language else ""
        question = state.question
        messages = state.messages if state.messages else []

        formatted_docs = self.embeddings.get_documents(question, top_k=20)
        state.documents = formatted_docs  # UPDATE IF A FILE IS UPLOADED FROM FRONTEND

        # format prompt to LLM
        input_dict = {
            "documents": formatted_docs,
            "language": language,
            "question": question,
            "messages": messages,
            "response": ""
        }
        print(input_dict)
        try:
            response = self.runnable.invoke(input_dict)
            response_text = response.content if hasattr(response, 'content') else response
        except Exception as e:
            logger.error(f"Model invocation failed: {e}")
            raise HTTPException(status_code=500, detail="Model invocation failed.")

        reasoning, answer = parse_response(response_text)
        user_message = HumanMessage(content=question)
        ai_response = AIMessage(content=f"### Thought Process:\n{reasoning}\n\n### Final Answer:\n{answer}")

        return {"messages": [user_message, ai_response]}


class State(BaseModel):
    language: Optional[str] = None
    messages: Optional[Annotated[Sequence[BaseMessage], add_messages]] = []  # UPDATES ARE APPENDED
    question: str
    documents: Optional[str] = None


# Create a single shared instance of LLMModelManager and Graph
shared_model_manager = LLMModelManager(dir_path='data', document_force_reload=True)
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", shared_model_manager.call_model)

# Add memory
memory = MemorySaver()
llm_app = workflow.compile(checkpointer=memory)


def get_model_manager():
    return shared_model_manager  # Return the shared instance


def get_llm_app():
    return llm_app


# --- FastAPI ---
async def run_inference(llm_app, state, config):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: llm_app.invoke(state, config))


@app.get("/")
async def read_index():
    return FileResponse('public/index.html')

# FastAPI endpoint for multi-user inference
@app.post("/chat/{thread_id}")
async def chat2(thread_id: str, state: State, background_tasks: BackgroundTasks,
                llm_app: StateGraph = Depends(get_llm_app)):
    logger.info(f"Received request for thread_id: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}
    response = await run_inference(llm_app, state, config)

    if "messages" in response and len(response["messages"]) > 0:
        latest_message = response["messages"][-1]

        if isinstance(latest_message, AIMessage):
            reasoning, answer = parse_response(latest_message.content)
            output = {"thought_process": reasoning, "answer": answer}
        else:
            output = {"thought_process": "", "answer": 'Error: No AI response found'}

    else:
        output = {"thought_process": "", "answer": 'Error: No message in response'}

    logger.debug(f"Response for thread_id {thread_id}: {response}")

    return output


if __name__ == "__main__":
    import uvicorn
   
    # Start FastAPI server
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
