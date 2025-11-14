"""
Web Search Handler - Integrazione con Tavily API
Fallback quando confidence è basso
"""

import os
from typing import List, Dict, Any, Optional
import requests
from datetime import datetime


class WebSearchHandler:
    """Gestisce ricerche web con Tavily API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "basic"  # "basic" or "advanced"
    ):
        """
        Args:
            api_key: Tavily API key (o usa env var TAVILY_API_KEY)
            max_results: Numero massimo di risultati
            search_depth: "basic" (veloce) o "advanced" (più completo)
        """
        # Prova a ottenere API key da parametro o environment
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Tavily API key non trovata!\n"
                "Passala al costruttore o imposta la variabile d'ambiente TAVILY_API_KEY"
            )
        
        self.max_results = max_results
        self.search_depth = search_depth
        self.api_url = "https://api.tavily.com/search"
        
        print(f"✓ WebSearchHandler inizializzato")
        print(f"  Search depth: {search_depth}")
        print(f"  Max results: {max_results}")
    
    def search(
        self,
        query: str,
        include_answer: bool = True,
        include_raw_content: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Esegue ricerca web con Tavily
        
        Args:
            query: Query di ricerca
            include_answer: Se True, Tavily genera una risposta riassuntiva
            include_raw_content: Se True, include contenuto completo delle pagine
            verbose: Stampa info di debug
            
        Returns:
            Dict con risultati della ricerca
        """
        
        if verbose:
            print(f"\n🌐 WEB SEARCH: '{query}'")
            print("="*60)
        
        # Payload per Tavily API
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_domains": [],  # Opzionale: limita a domini specifici
            "exclude_domains": []   # Opzionale: escludi domini
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            results = response.json()
            
            if verbose:
                num_results = len(results.get('results', []))
                print(f"✓ Trovati {num_results} risultati")
                
                if include_answer and 'answer' in results:
                    print(f"✓ Risposta generata da Tavily")
                
                print("="*60)
            
            return results
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Errore nella ricerca web: {e}")
            return {
                "results": [],
                "answer": None,
                "error": str(e)
            }
    
    def format_results(
        self,
        search_results: Dict[str, Any],
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Formatta risultati Tavily in formato standard
        
        Args:
            search_results: Output da self.search()
            max_results: Numero massimo di risultati da formattare
            
        Returns:
            Lista di risultati formattati
        """
        formatted = []
        
        raw_results = search_results.get('results', [])
        
        for result in raw_results[:max_results]:
            formatted.append({
                'title': result.get('title', 'No title'),
                'url': result.get('url', ''),
                'content': result.get('content', ''),
                'score': result.get('score', 0.0),
                'published_date': result.get('published_date', None)
            })
        
        return formatted
    
    def create_web_context(
        self,
        search_results: Dict[str, Any],
        max_results: int = 3
    ) -> str:
        """
        Crea context assemblato dai risultati web (simile a PDF chunks)
        
        Args:
            search_results: Output da self.search()
            max_results: Numero di risultati da includere
            
        Returns:
            Context formattato per LLM
        """
        results = self.format_results(search_results, max_results)
        
        if not results:
            return "No web results found."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            # Formatta ogni risultato
            context_part = (
                f"[Web Source {i}: {result['title']}]\n"
                f"URL: {result['url']}\n"
                f"Relevance Score: {result['score']:.2f}\n"
                f"Content:\n{result['content']}\n"
            )
            context_parts.append(context_part)
        
        # Unisci con separatori
        assembled = "\n" + "="*60 + "\n".join(context_parts)
        
        return assembled
    
    def get_tavily_answer(self, search_results: Dict[str, Any]) -> Optional[str]:
        """
        Estrae la risposta generata da Tavily (se presente)
        
        Args:
            search_results: Output da self.search()
            
        Returns:
            Risposta di Tavily o None
        """
        return search_results.get('answer')
    
    def test_connection(self) -> bool:
        """Test veloce della connessione API"""
        print("\n🧪 Testing Tavily API connection...")
        
        try:
            results = self.search(
                query="Python programming",
                include_answer=False,
                verbose=False
            )
            
            if 'error' in results:
                print(f"❌ API Error: {results['error']}")
                return False
            
            num_results = len(results.get('results', []))
            
            if num_results > 0:
                print(f"✅ Tavily API funzionante! ({num_results} risultati)")
                return True
            else:
                print("⚠️  API risponde ma nessun risultato")
                return False
        
        except Exception as e:
            print(f"❌ Test fallito: {e}")
            return False


# ============= ESEMPIO D'USO =============

if __name__ == "__main__":
    print("="*70)
    print("STEP 5: WEB SEARCH HANDLER TEST")
    print("="*70)
    
    # Inizializza (inserisci la tua API key qui o usa environment variable)
    # Opzione 1: Passa direttamente
    API_KEY = "tvly-YOUR_API_KEY_HERE"  # ← SOSTITUISCI CON LA TUA KEY
    
    # Opzione 2: Usa environment variable
    # Esegui prima in terminal: $env:TAVILY_API_KEY="tvly-your-key"
    # API_KEY = None  # Leggerà da environment
    
    try:
        web_search = WebSearchHandler(
            api_key=API_KEY,
            max_results=5,
            search_depth="basic"
        )
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\nPer usare questo modulo:")
        print("1. Vai su https://tavily.com")
        print("2. Ottieni la tua API key")
        print("3. Sostituisci 'tvly-YOUR_API_KEY_HERE' nel codice")
        print("   OPPURE")
        print("4. Imposta env variable: $env:TAVILY_API_KEY='tvly-your-key'")
        exit(1)
    
    # Test 1: Verifica connessione
    print("\n" + "="*70)
    print("TEST 1: Connection Test")
    print("="*70)
    web_search.test_connection()
    
    # Test 2: Ricerca base
    print("\n" + "="*70)
    print("TEST 2: Basic Search")
    print("="*70)
    
    query = "What is the latest Python version in 2024?"
    print(f"\nQuery: {query}")
    
    results = web_search.search(
        query=query,
        include_answer=True,
        verbose=True
    )
    
    # Mostra risposta di Tavily
    tavily_answer = web_search.get_tavily_answer(results)
    if tavily_answer:
        print(f"\n📝 TAVILY ANSWER:")
        print("="*70)
        print(tavily_answer)
        print("="*70)
    
    # Mostra risultati formattati
    formatted = web_search.format_results(results, max_results=3)
    print(f"\n🔍 TOP {len(formatted)} RESULTS:")
    print("="*70)
    
    for i, result in enumerate(formatted, 1):
        print(f"\n[{i}] {result['title']}")
        print(f"    Score: {result['score']:.2f}")
        print(f"    URL: {result['url']}")
        print(f"    Content: {result['content'][:200]}...")
    
    # Test 3: Context assembly (per LLM)
    print("\n" + "="*70)
    print("TEST 3: Context Assembly for LLM")
    print("="*70)
    
    context = web_search.create_web_context(results, max_results=2)
    print(f"\nAssembled Context ({len(context)} chars):")
    print(context[:500] + "...\n")
    
    print("="*70)
    print("✅ WEB SEARCH HANDLER TEST COMPLETATO!")
    print("="*70)