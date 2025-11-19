"""
LLM Handler - Gestisce comunicazione con Ollama (Mistral)
"""

import requests
import json
from typing import Optional
import time


class LLMHandler:
    """Wrapper per Ollama API con prompt engineering per RAG"""
    
    def __init__(
        self,
        model_name: str = "mistral:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        timeout: int = 120
    ):
        """
        Args:
            model_name: Nome modello Ollama (es: "mistral:7b")
            base_url: URL dell'API Ollama (default locale)
            temperature: Creatività (0=deterministico, 1=creativo)
            timeout: Timeout in secondi per la richiesta
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.api_url = f"{base_url}/api/generate"
        
        # Verifica che Ollama sia attivo
        self._check_ollama_running()
    
    def _check_ollama_running(self):
        """Verifica che il servizio Ollama sia attivo"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Ollama service attivo")
                
                # Verifica che il modello sia disponibile
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name not in model_names:
                    print(f"⚠️  Modello '{self.model_name}' non trovato")
                    print(f"   Modelli disponibili: {model_names}")
                    print(f"   Scaricalo con: ollama pull {self.model_name}")
                else:
                    print(f"✓ Modello {self.model_name} disponibile")
        
        except requests.exceptions.ConnectionError:
            print("❌ Ollama non raggiungibile!")
            print("   Assicurati che Ollama sia in esecuzione.")
            print("   Apri un terminale e scrivi: ollama serve")
            raise ConnectionError("Ollama service non raggiungibile")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        stream: bool = False,
        verbose: bool = False
    ) -> str:
        """
        Genera risposta dal modello
        
        Args:
            prompt: Prompt principale (query + context)
            system_prompt: System prompt (comportamento del modello)
            max_tokens: Numero massimo di token da generare
            stream: Se True, stampa risposta in streaming
            verbose: Stampa info di debug
            
        Returns:
            Risposta generata dal modello
        """
        
        # Costruisci prompt completo
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Payload per Ollama API
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            }
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🤖 LLM REQUEST:")
            print(f"{'='*60}")
            print(f"Model: {self.model_name}")
            print(f"Temperature: {self.temperature}")
            print(f"Max tokens: {max_tokens}")
            print(f"Prompt length: {len(full_prompt)} chars")
            print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                # Streaming response (stampa man mano)
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        chunk = json_response.get('response', '')
                        full_response += chunk
                        print(chunk, end='', flush=True)
                print()  # Newline alla fine
                return full_response
            
            else:
                # Risposta completa
                result = response.json()
                generated_text = result.get('response', '')
                
                elapsed_time = time.time() - start_time
                
                if verbose:
                    total_tokens = result.get('eval_count', 0)
                    tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
                    
                    print(f"\n{'='*60}")
                    print(f"🤖 LLM RESPONSE:")
                    print(f"{'='*60}")
                    print(f"Generation time: {elapsed_time:.2f}s")
                    print(f"Tokens generated: {total_tokens}")
                    print(f"Speed: {tokens_per_sec:.1f} tokens/sec")
                    print(f"Response length: {len(generated_text)} chars")
                    print(f"{'='*60}\n")
                
                return generated_text
        
        except requests.exceptions.Timeout:
            return f"⚠️ Timeout: Il modello ha impiegato troppo tempo (>{self.timeout}s)"
        
        except requests.exceptions.RequestException as e:
            return f"❌ Errore nella richiesta: {str(e)}"
    
    def generate_rag_response(
        self,
        query: str,
        context: str,
        verbose: bool = False
    ) -> str:
        """
        Genera risposta usando RAG (Retrieval Augmented Generation)
        
        Args:
            query: Domanda dell'utente
            context: Context assemblato dai chunks
            verbose: Info di debug
            
        Returns:
            Risposta generata basata sul context
        """
        
        system_prompt = """You are a helpful RAG (Retrieval Augmented Generation) assistant.

Your task is to answer questions based ONLY on the provided context from documents.

RULES:
1. Use ONLY information from the context provided
2. If the context doesn't contain the answer, say: "I cannot find this information in the provided documents."
3. Cite the source when possible (mention page numbers)
4. Be concise and accurate
5. Do NOT make up information or use external knowledge
6. If unsure, express uncertainty clearly"""
        
        prompt = f"""Context from documents:
{context}

User Question: {query}

Answer (based only on the context above):"""
        
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=500,
            verbose=verbose
        )
        
        return response.strip()
    
    def test_model(self):
        """Test veloce del modello"""
        print(f"\n🧪 Testing {self.model_name}...")
        
        test_prompt = "Say 'Hello! I am working correctly.' in one sentence."
        response = self.generate(test_prompt, verbose=False)
        
        print(f"Response: {response}\n")
        
        if response and len(response) > 10:
            print("✅ Modello funzionante!")
            return True
        else:
            print("❌ Modello non risponde correttamente")
            return False