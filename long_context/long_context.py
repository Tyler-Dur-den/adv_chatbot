from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from domain_aware_rag.config import settings

llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0
    )
