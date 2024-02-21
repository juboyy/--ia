import gradio as gr
import os
import time
import arxiv
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

   #Define função de UI para processar os artigos do Arxiv
def process_papers(query, question_text):

    # Pesquisar Arxiv por artigos e baixa-los
    dirPath = "arxiv_papers"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    # Pesquisar por artigos que contenham "LLM"
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=15,
        sort_order = arxiv.SortOrder.Descending)

    # Baixar e salvar os artigos
    for result in client.results(search):
        while True:
            try:
                result.download_pdf(dirpath= dirPath)
                print(result)
                print(f"-> Paper id {result.get_short_id()} com título '{result.title}' foi baixado.")
                break
            except (FileNotFoundError, ConnectionResetError) as e:
                print("Erro Ocorreu:", e)
                time.sleep(5)

    # Carregar os artigos
    papers = []
    loader = DirectoryLoader(dirPath, glob="./*.pdf", loader_cls=PyPDFLoader)
    try:
        papers = loader.load()
    except Exception as e:
        print(f"Erro ao carregar o arxiv: {e}")
    print("total de páginas carregadas:", len(papers))

    # Concatenar todas as paginas em um único string
    full_text = ""
    for paper in papers:
        full_text += paper.page_content

    # Remove linhas vazias e adiciona em uma unica linha
    full_text = " ".join(line for line in full_text.splitlines() if line)
    print("Total de caracteres no texto concatenado:", len(full_text))

    # Dividir o texto em parágrafos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([full_text])

    # Criar Qdrant Vector Store e armazenar os embeddings

    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding=GPT4AllEmbeddings(),
        path="./tmp/local_qdrant",
        collection_name="arxiv_papers",
    )
    retriever = qdrant.as_retriever()

    # Defina o template do prompt
    template = """Resposta baseada no contexto:
    {context}

    Pergunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Inicialize Ollama LLM
    ollama_llm = "llama2:7b-chat"
    model = ChatOllama(model=ollama_llm)

    # Defina a Cadeia de execução
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

    class Question(BaseModel):
        __root__: str
    
    chain = chain.with_types(input_type=Question)
    result = chain.invoke(question_text)
    return result

iface = gr.Interface(
    fn=process_papers,
    inputs=["text", "text"],
    outputs="text",
    description="Enter a search query and a question to process arXiv papers."
)

iface.launch()
