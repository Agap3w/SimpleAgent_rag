"""
Main.py - Sistema RAG Completo
Entry point principale per il progetto
"""

import sys
from pathlib import Path
from typing import Optional

# Import dei nostri moduli
from ingestion import PDFIngestion
from embeddings import EmbeddingSystem
from query_system import QuerySystem
from llm_handler import LLMHandler
from rag_pipeline import RAGPipeline

import os


class RAGSystem:
    """Sistema RAG completo - orchestrator principale"""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "pdf_knowledge_base",
        llm_model: str = "mistral:7b",
        tavily_api_key: Optional[str] = None,
        enable_web_search: bool = True
    ):
        """
        Inizializza il sistema RAG completo
        
        Args:
            persist_directory: Directory per ChromaDB
            collection_name: Nome collection
            llm_model: Modello Ollama da usare
            tavily_api_key: API key per Tavily (web search)
            enable_web_search: Se True, abilita web fallback
        """
        print("="*70)
        print("🚀 INITIALIZING RAG SYSTEM")
        print("="*70)
        
        # 1. PDF Ingestion
        print("\n[1/5] Setting up PDF Ingestion...")
        self.pdf_ingestion = PDFIngestion(
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=100
        )
        print("   ✓ PDF Ingestion ready")
        
        # 2. Embedding System
        print("\n[2/5] Setting up Embedding System...")
        self.embedding_system = EmbeddingSystem(
            model_name="all-MiniLM-L6-v2",
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # 3. Query System
        print("\n[3/5] Setting up Query System...")
        self.query_system = QuerySystem(
            embedding_system=self.embedding_system,
            confidence_threshold_high=0.6,
            confidence_threshold_low=0.4
        )
        print("   ✓ Query System ready")
        
        # 4. LLM Handler
        print("\n[4/5] Setting up LLM Handler...")
        self.llm_handler = LLMHandler(
            model_name=llm_model,
            temperature=0.3
        )
        
        # 5. Web Search Handler (opzionale)
        print("\n[5/5] Setting up Web Search...")
        web_search_handler = None
        
        if enable_web_search:
            try:
                from web_search_handler import WebSearchHandler
                web_search_handler = WebSearchHandler(
                    api_key=tavily_api_key,
                    max_results=5,
                    search_depth="basic"
                )
            except (ValueError, ImportError) as e:
                print(f"   ⚠️  Web search non disponibile: {e}")
                print("   Continuando senza web fallback...")
        else:
            print("   ⚠️  Web search disabilitato")
        
        # 6. RAG Pipeline
        self.rag_pipeline = RAGPipeline(
            query_system=self.query_system,
            llm_handler=self.llm_handler,
            web_search_handler=web_search_handler,
            enable_web_fallback=enable_web_search and web_search_handler is not None
        )
        
        print("\n" + "="*70)
        print("✅ RAG SYSTEM READY!")
        print("="*70 + "\n")
    
    def ingest_pdf(self, pdf_path: str):
        """
        Ingests un PDF nel knowledge base
        
        Args:
            pdf_path: Path al file PDF
        """
        print(f"\n📥 INGESTING PDF: {pdf_path}")
        print("="*70)
        
        # Step 1: Process PDF
        chunks = self.pdf_ingestion.process_pdf(pdf_path)
        
        # Step 2: Add to vector database
        self.embedding_system.add_chunks_to_db(chunks, batch_size=64)
        
        print("="*70)
        print(f"✅ PDF ingestion completato!")
        print(f"   Total documents in DB: {self.embedding_system.collection.count()}")
        print("="*70 + "\n")
    
    def ingest_directory(self, directory_path: str):
        """
        Ingests tutti i PDF in una directory
        
        Args:
            directory_path: Path alla directory
        """
        print(f"\n📁 INGESTING DIRECTORY: {directory_path}")
        print("="*70)
        
        # Step 1: Process all PDFs
        all_chunks = self.pdf_ingestion.process_directory(directory_path)
        
        if all_chunks:
            # Step 2: Add to vector database
            self.embedding_system.add_chunks_to_db(all_chunks, batch_size=64)
        
        print("="*70)
        print(f"✅ Directory ingestion completato!")
        print(f"   Total documents in DB: {self.embedding_system.collection.count()}")
        print("="*70 + "\n")
    
    def query(self, question: str, verbose: bool = True):
        """
        Query il sistema RAG
        
        Args:
            question: Domanda dell'utente
            verbose: Stampa info dettagliate
            
        Returns:
            RAGResponse object
        """
        return self.rag_pipeline.process_query(question, verbose=verbose)
    
    def interactive_mode(self):
        """Modalità interattiva - chat continua"""
        print("\n" + "="*70)
        print("💬 INTERACTIVE MODE")
        print("="*70)
        print("Ask questions about your documents!")
        print("Commands: 'exit' or 'quit' to stop\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Process query
                print()  # Newline for better formatting
                response = self.query(user_input, verbose=True)
                
                print()  # Newline before next input
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


# ============= MAIN ENTRY POINT =============

def main():
    """Main function - entry point"""
    
    # Ottieni Tavily API key
    print("\n🔑 Web Search Configuration:")
    print("="*70)
    
    # Prova a leggere da environment variable
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if tavily_key:
        print("✓ Tavily API key trovata in environment variable")
        enable_web = True
    else:
        print("⚠️  Tavily API key non trovata in environment")
        user_input = input("\nVuoi inserire la API key ora? (y/n): ").strip().lower()
        
        if user_input == 'y':
            tavily_key = input("Inserisci Tavily API key: ").strip()
            enable_web = True if tavily_key else False
        else:
            print("Continuando SENZA web search...")
            tavily_key = None
            enable_web = False
    
    # Initialize RAG System
    rag = RAGSystem(
        persist_directory="./chroma_db",
        collection_name="pdf_knowledge_base",
        llm_model="mistral:7b",
        tavily_api_key=tavily_key,
        enable_web_search=enable_web
    )
    
    # Check if database is empty
    doc_count = rag.embedding_system.collection.count()
    
    if doc_count == 0:
        print("⚠️  Knowledge base is empty!")
        print("   Please ingest PDFs first.\n")
        
        # Example: ingest a PDF
        pdf_path = input("Enter PDF path (or press Enter to skip): ").strip()
        
        if pdf_path and Path(pdf_path).exists():
            rag.ingest_pdf(pdf_path)
        else:
            print("\n⚠️  No PDF ingested. Using empty database.\n")
    
    # Choose mode
    print("\n" + "="*70)
    print("SELECT MODE:")
    print("="*70)
    print("1. Interactive Chat Mode")
    print("2. Single Query Test")
    print("3. Batch Test Queries")
    print("="*70)
    
    choice = input("\nYour choice (1-3): ").strip()
    
    if choice == "1":
        # Interactive mode
        rag.interactive_mode()
    
    elif choice == "2":
        # Single query
        question = input("\nEnter your question: ").strip()
        if question:
            rag.query(question, verbose=True)
    
    elif choice == "3":
        # Batch test
        test_queries = [
            "What are Python dictionaries?",
            "How do I use for loops in Python?",
            "Explain list comprehensions",
            "What is exception handling?",
        ]
        
        print(f"\n🧪 Running {len(test_queries)} test queries...\n")
        
        for i, q in enumerate(test_queries, 1):
            print(f"\n{'#'*70}")
            print(f"QUERY {i}/{len(test_queries)}")
            print(f"{'#'*70}")
            
            rag.query(q, verbose=True)
            
            if i < len(test_queries):
                input("\n⏸️  Press Enter for next query...")
    
    else:
        print("Invalid choice. Exiting.")
    
    print("\n" + "="*70)
    print("✅ SESSION COMPLETED")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)