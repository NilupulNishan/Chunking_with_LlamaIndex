"""
Embeddings module for Azure OpenAI integration.
"""
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings
from config import settings


class EmbeddingsManager:
    """Manages LLM and embedding model initialization."""
    
    def __init__(self):
        """Initialize Azure OpenAI models."""
        settings.validate_config()
        
        # Initialize LLM
        self.llm = AzureOpenAI(
            model="gpt-4o",
            deployment_name=settings.AZURE_CHAT_DEPLOYMENT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.OPENAI_API_VERSION,
            temperature=0.1,
        )
        
        # Initialize embedding model
        self.embed_model = AzureOpenAIEmbedding(
            model="text-embedding-3-large",
            deployment_name=settings.AZURE_EMBEDDING_DEPLOYMENT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.OPENAI_API_VERSION,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
        
        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
        print("✓ Azure OpenAI models initialized")
    
    def get_llm(self):
        """Get the language model."""
        return self.llm
    
    def get_embed_model(self):
        """Get the embedding model."""
        return self.embed_model


if __name__ == "__main__":
    # Test embeddings manager
    manager = EmbeddingsManager()
    print(f"LLM Model: {manager.llm.model}")
    print(f"Embedding Model: {manager.embed_model.model}")
    print(f"Embedding Dimensions: {settings.EMBEDDING_DIMENSIONS}")