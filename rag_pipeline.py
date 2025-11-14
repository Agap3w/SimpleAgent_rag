"""
RAG Pipeline - Integrazione completa di tutto il sistema
Query → Confidence → LLM/Web Fallback → Response
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from query_system import QuerySystem, QueryResult
from llm_handler import LLMHandler
import time


@dataclass
class RAGResponse:
    """Risposta completa del sistema RAG"""
    query: str
    answer: str
    confidence: float
    confidence_category: str
    source_type: str  # "pdf" o "web"
    sources: list
    response_time: float
    used_web_fallback: bool


class RAGPipeline:
    """
    Pipeline completo RAG:
    1. Riceve query utente
    2. Cerca nel knowledge base (PDF)
    3. Calcola confidence
    4. Se confidence OK → usa LLM con PDF context
    5. Se confidence BASSO → usa web search (TODO Step 5)
    6. Genera risposta finale con citazioni
    """
    
    def __init__(
        self,
        query_system: QuerySystem,
        llm_handler: LLMHandler,
        enable_web_fallback: bool = False  # Per ora False, implementeremo Step 5
    ):
        """
        Args:
            query_system: Sistema di query con confidence
            llm_handler: Handler per LLM (Mistral)
            enable_web_fallback: Se True, usa web search quando confidence basso
        """
        self.query_system = query_system
        self.llm_handler = llm_handler
        self.enable_web_fallback = enable_web_fallback
        
        print("✓ RAG Pipeline inizializzato")
        if not enable_web_fallback:
            print("  ⚠️  Web fallback disabilitato (Step 5 non ancora implementato)")
    
    def process_query(
        self,
        query: str,
        verbose: bool = True
    ) -> RAGResponse:
        """
        Processa query completa attraverso la pipeline
        
        Args:
            query: Domanda dell'utente
            verbose: Stampa info durante il processo
            
        Returns:
            RAGResponse con risposta e metadata
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🔍 PROCESSING QUERY: {query}")
            print(f"{'='*70}")
        
        # STEP 1: Query knowledge base + confidence scoring
        if verbose:
            print("\n[1/3] Searching knowledge base...")
        
        query_result = self.query_system.query(query, verbose=False)
        
        if verbose:
            print(f"   ✓ Confidence: {query_result.confidence:.3f} ({query_result.confidence_category})")
            print(f"   ✓ Retrieved {len(query_result.retrieved_chunks)} chunks")
        
        # STEP 2: Decide source (PDF vs Web)
        if query_result.should_use_web_fallback and self.enable_web_fallback:
            if verbose:
                print(f"\n[2/3] 🌐 Using WEB SEARCH (low confidence)...")
            
            # TODO: Implement web search in Step 5
            answer = "⚠️ Web fallback not yet implemented. Please enable it in Step 5."
            source_type = "web"
            sources = []
        
        else:
            if verbose:
                print(f"\n[2/3] 📄 Using PDF CONTENT...")
            
            # STEP 3: Generate answer with LLM
            if verbose:
                print(f"\n[3/3] 🤖 Generating answer with LLM...")
            
            answer = self.llm_handler.generate_rag_response(
                query=query,
                context=query_result.assembled_context,
                verbose=verbose
            )
            
            source_type = "pdf"
            sources = [
                {
                    'source': chunk['metadata']['source'],
                    'page': chunk['metadata']['page'],
                    'similarity': chunk['similarity_score']
                }
                for chunk in query_result.retrieved_chunks[:3]  # Top 3 sources
            ]
        
        response_time = time.time() - start_time
        
        # Construct final response
        rag_response = RAGResponse(
            query=query,
            answer=answer,
            confidence=query_result.confidence,
            confidence_category=query_result.confidence_category,
            source_type=source_type,
            sources=sources,
            response_time=response_time,
            used_web_fallback=query_result.should_use_web_fallback
        )
        
        if verbose:
            self._print_response(rag_response)
        
        return rag_response
    
    def _print_response(self, response: RAGResponse):
        """Stampa risposta formattata"""
        print(f"\n{'='*70}")
        print(f"✅ RESPONSE GENERATED")
        print(f"{'='*70}")
        
        print(f"\n📝 ANSWER:")
        print(f"{response.answer}")
        
        print(f"\n📊 METADATA:")
        print(f"   Confidence: {response.confidence:.3f} ({response.confidence_category})")
        print(f"   Source Type: {response.source_type.upper()}")
        print(f"   Response Time: {response.response_time:.2f}s")
        print(f"   Used Fallback: {'Yes' if response.used_web_fallback else 'No'}")
        
        if response.sources:
            print(f"\n📚 SOURCES:")
            for i, source in enumerate(response.sources, 1):
                print(f"   [{i}] {source['source']} (Page {source['page']}, Relevance: {source['similarity']:.2f})")
        
        print(f"\n{'='*70}\n")


# ============= ESEMPIO D'USO COMPLETO =============

if __name__ == "__main__":
    from embeddings import EmbeddingSystem
    from query_system import QuerySystem
    from llm_handler import LLMHandler
    
    print("="*70)
    print("STEP 4: RAG PIPELINE - FULL INTEGRATION TEST")
    print("="*70)
    
    # Setup components
    print("\n🔧 Initializing components...")
    
    # 1. Embedding System (assume DB already populated)
    embedding_system = EmbeddingSystem(
        model_name="all-MiniLM-L6-v2",
        collection_name="python_tutorial",
        persist_directory="./chroma_db"
    )
    
    # 2. Query System
    query_system = QuerySystem(
        embedding_system=embedding_system,
        confidence_threshold_high=0.6,
        confidence_threshold_low=0.4
    )
    
    # 3. LLM Handler
    llm_handler = LLMHandler(
        model_name="mistral:7b",
        temperature=0.3
    )
    
    # 4. RAG Pipeline
    rag_pipeline = RAGPipeline(
        query_system=query_system,
        llm_handler=llm_handler,
        enable_web_fallback=False  # Step 5 not implemented yet
    )
    
    print("\n✅ All components ready!\n")
    
    # Test queries
    test_queries = [
        "What are Python dictionaries and how do I use them?",
        "Explain list comprehensions with examples",
        "How do I handle exceptions in Python?",
    ]
    
    print("="*70)
    print("🧪 TESTING RAG PIPELINE")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*70}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"{'#'*70}")
        
        response = rag_pipeline.process_query(query, verbose=True)
        
        if i < len(test_queries):
            input("\n⏸️  Press Enter for next query...")
    
    print("\n" + "="*70)
    print("✅ RAG PIPELINE TEST COMPLETATO!")
    print("="*70)
    print("\nIl sistema ora:")
    print("  ✅ Riceve query utente")
    print("  ✅ Cerca nel knowledge base PDF")
    print("  ✅ Calcola confidence score")
    print("  ✅ Genera risposte con LLM (Mistral)")
    print("  ✅ Fornisce citazioni accurate")
    print("  ⏳ Web fallback (Step 5 - TODO)")
    print("\n🎉 Step 4 COMPLETATO!")