"""
RAG Pipeline - Integrazione completa di tutto il sistema
Query → Confidence → LLM/Web Fallback → Response
UPDATED: Con Web Search Integration (Step 5)
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from query_system import QuerySystem, QueryResult
from llm_handler import LLMHandler
from web_search_handler import WebSearchHandler
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
    web_search_results: Optional[Dict] = None  # Risultati raw da Tavily


class RAGPipeline:
    """
    Pipeline completo RAG con Web Fallback:
    1. Riceve query utente
    2. Cerca nel knowledge base (PDF)
    3. Calcola confidence
    4. Se confidence OK → usa LLM con PDF context
    5. Se confidence BASSO → usa web search
    6. Genera risposta finale con citazioni
    """
    
    def __init__(
        self,
        query_system: QuerySystem,
        llm_handler: LLMHandler,
        web_search_handler: Optional[WebSearchHandler] = None,
        enable_web_fallback: bool = True
    ):
        """
        Args:
            query_system: Sistema di query con confidence
            llm_handler: Handler per LLM (Mistral)
            web_search_handler: Handler per web search (Tavily)
            enable_web_fallback: Se True, usa web search quando confidence basso
        """
        self.query_system = query_system
        self.llm_handler = llm_handler
        self.web_search_handler = web_search_handler
        self.enable_web_fallback = enable_web_fallback
        
        print("✓ RAG Pipeline inizializzato")
        
        if enable_web_fallback:
            if web_search_handler:
                print("  ✅ Web fallback ABILITATO (Tavily API)")
            else:
                print("  ⚠️  Web fallback abilitato ma WebSearchHandler non fornito!")
                self.enable_web_fallback = False
        else:
            print("  ⚠️  Web fallback disabilitato")
    
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
            print("\n[1/4] Searching knowledge base...")
        
        query_result = self.query_system.query(query, verbose=False)
        
        if verbose:
            print(f"   ✓ Confidence: {query_result.confidence:.3f} ({query_result.confidence_category})")
            print(f"   ✓ Retrieved {len(query_result.retrieved_chunks)} chunks")
        
        # STEP 2: Decide source (PDF vs Web)
        web_search_results = None
        
        if query_result.should_use_web_fallback and self.enable_web_fallback:
            if verbose:
                print(f"\n[2/4] 🌐 LOW CONFIDENCE - Using WEB SEARCH...")
            
            # Esegui web search
            web_search_results = self.web_search_handler.search(
                query=query,
                include_answer=True,
                verbose=verbose
            )
            
            # Crea context dai risultati web
            web_context = self.web_search_handler.create_web_context(
                web_search_results,
                max_results=3
            )
            
            if verbose:
                print(f"\n[3/4] 🤖 Generating answer from WEB results...")
            
            # Genera risposta usando context web
            answer = self._generate_web_answer(
                query=query,
                web_context=web_context,
                web_results=web_search_results,
                verbose=verbose
            )
            
            source_type = "web"
            
            # Format sources da web
            formatted_web_results = self.web_search_handler.format_results(
                web_search_results,
                max_results=3
            )
            sources = [
                {
                    'title': r['title'],
                    'url': r['url'],
                    'score': r['score']
                }
                for r in formatted_web_results
            ]
        
        else:
            if verbose:
                print(f"\n[2/4] 📄 GOOD CONFIDENCE - Using PDF CONTENT...")
            
            # STEP 3: Generate answer with LLM da PDF
            if verbose:
                print(f"\n[3/4] 🤖 Generating answer from PDF...")
            
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
                for chunk in query_result.retrieved_chunks[:3]
            ]
        
        response_time = time.time() - start_time
        
        if verbose:
            print(f"\n[4/4] ✅ Response completed in {response_time:.2f}s")
        
        # Construct final response
        rag_response = RAGResponse(
            query=query,
            answer=answer,
            confidence=query_result.confidence,
            confidence_category=query_result.confidence_category,
            source_type=source_type,
            sources=sources,
            response_time=response_time,
            used_web_fallback=query_result.should_use_web_fallback,
            web_search_results=web_search_results
        )
        
        if verbose:
            self._print_response(rag_response)
        
        return rag_response
    
    def _generate_web_answer(
        self,
        query: str,
        web_context: str,
        web_results: Dict[str, Any],
        verbose: bool = False
    ) -> str:
        """
        Genera risposta usando risultati web
        
        Usa la risposta di Tavily se disponibile, altrimenti genera con LLM
        """
        # Opzione 1: Usa risposta diretta di Tavily (se disponibile)
        tavily_answer = self.web_search_handler.get_tavily_answer(web_results)
        
        if tavily_answer and len(tavily_answer) > 50:
            # Tavily ha già generato una buona risposta
            return f"{tavily_answer}\n\n(Answer generated from web search results)"
        
        # Opzione 2: Genera con LLM usando context web
        system_prompt = """You are a helpful assistant answering questions based on web search results.

RULES:
1. Use ONLY information from the provided web search results
2. Cite sources by mentioning the website or article title
3. Be concise and accurate
4. If the results don't fully answer the question, say so
5. Indicate that information comes from web search"""
        
        prompt = f"""Web Search Results:
{web_context}

User Question: {query}

Answer (based on the web search results above):"""
        
        response = self.llm_handler.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=500,
            verbose=verbose
        )
        
        return response.strip()
    
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
            
            if response.source_type == "pdf":
                for i, source in enumerate(response.sources, 1):
                    print(f"   [{i}] {source['source']} (Page {source['page']}, Relevance: {source['similarity']:.2f})")
            else:  # web
                for i, source in enumerate(response.sources, 1):
                    print(f"   [{i}] {source['title']}")
                    print(f"       URL: {source['url']}")
                    print(f"       Score: {source['score']:.2f}")
        
        print(f"\n{'='*70}\n")


# ============= ESEMPIO D'USO COMPLETO =============

if __name__ == "__main__":
    from embeddings import EmbeddingSystem
    from query_system import QuerySystem
    from llm_handler import LLMHandler
    from web_search_handler import WebSearchHandler
    
    print("="*70)
    print("STEP 5: RAG PIPELINE WITH WEB FALLBACK TEST")
    print("="*70)
    
    # Setup components
    print("\n🔧 Initializing components...")
    
    # 1. Embedding System
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
    
    # 4. Web Search Handler
    print("\n🌐 Setting up Web Search (Tavily)...")
    TAVILY_API_KEY = "tvly-YOUR_KEY_HERE"  # ← INSERISCI LA TUA KEY
    
    try:
        web_search = WebSearchHandler(
            api_key=TAVILY_API_KEY,
            max_results=5,
            search_depth="basic"
        )
    except ValueError as e:
        print(f"\n⚠️  {e}")
        print("Continuando SENZA web fallback...")
        web_search = None
    
    # 5. RAG Pipeline
    rag_pipeline = RAGPipeline(
        query_system=query_system,
        llm_handler=llm_handler,
        web_search_handler=web_search,
        enable_web_fallback=True if web_search else False
    )
    
    print("\n✅ All components ready!\n")
    
    # Test queries - mix di PDF e Web
    test_queries = [
        # Query che dovrebbe usare PDF (alta confidence)
        "What are Python dictionaries?",
        
        # Query che dovrebbe usare WEB (bassa confidence - info recente)
        "What is the latest Python version released in 2024?",
        
        # Query fuori dal PDF (dovrebbe usare web)
        "Who is the current CEO of OpenAI?",
    ]
    
    print("="*70)
    print("🧪 TESTING RAG PIPELINE WITH WEB FALLBACK")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*70}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"{'#'*70}")
        
        response = rag_pipeline.process_query(query, verbose=True)
        
        if i < len(test_queries):
            input("\n⏸️  Press Enter for next query...")
    
    print("\n" + "="*70)
    print("✅ RAG PIPELINE WITH WEB FALLBACK TEST COMPLETATO!")
    print("="*70)
    print("\nIl sistema ora:")
    print("  ✅ Riceve query utente")
    print("  ✅ Cerca nel knowledge base PDF")
    print("  ✅ Calcola confidence score")
    print("  ✅ Genera risposte con LLM (Mistral)")
    print("  ✅ Web search fallback quando confidence basso")
    print("  ✅ Fornisce citazioni da PDF e Web")
    print("\n🎉 Step 5 COMPLETATO!")