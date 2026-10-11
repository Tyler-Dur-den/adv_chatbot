from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    HF_TOKEN: str
    GOOGLE_API_KEY: str  
    VECTOR_DB_PATH: str = "./chroma_db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",extra="ignore")

settings = Settings()

class RAGConfig:
    MAX_CONTEXT_CHARS = 12000
    
    DOC_CHUNK_CONFIG = {
        "Financial Report": {"chunk_size": 1000, "overlap": 100, "k": 4},
        "Legal Contract": {"chunk_size": 800, "overlap": 200, "k": 3},
        "Academic Paper": {"use_semantic": True, "k": 3, "fetch_k": 20},
        "default": {"chunk_size": 500, "overlap": 50, "k": 3}
    }

