"""
Query System with Confidence Scoring & Decision Logic
Integra search, confidence calculation, e context assembly
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from embeddings import EmbeddingSystem


@dataclass
class QueryResult:
    """Risultato strutturato di una query"""
    query: str
    confidence: float
    should_use_web_fallback: bool
    retrieved_chunks: List[Dict[str, Any]]
    assembled_context: str
    confidence_category: str  # "high", "medium", "low"
    top_similarity: float
    avg_similarity: float


class QuerySystem:
    """
    Sistema di query con confidence scoring e decision logic
    """
    
    def __init__(
        self,
        embedding_system: EmbeddingSystem,
        confidence_threshold_high: float = 0.6,
        confidence_threshold_low: float = 0.4,
        n_results: int = 5,
        context_max_chunks: int = 3
    ):
        """
        Args:
            embedding_system: Sistema di embeddings già inizializzato
            confidence_threshold_high: Soglia per "alta confidenza" (>= 0.6)
            confidence_threshold_low: Soglia per "bassa confidenza" (< 0.4)
            n_results: Numero di chunks da recuperare
            context_max_chunks: Max chunks da usare per context assembly
        """
        self.embedding_system = embedding_system
        self.confidence_threshold_high = confidence_threshold_high
        self.confidence_threshold_low = confidence_threshold_low
        self.n_results = n_results
        self.context_max_chunks = context_max_chunks
    
    def calculate_aggregate_confidence(
        self,
        similarity_scores: List[float]
    ) -> Tuple[float, str]:
        """
        Calcola confidence aggregato dai similarity scores
        
        Strategia: weighted average che privilegia i top results
        - Top result ha peso maggiore
        - Se top result è basso, penalizza tutto
        
        Args:
            similarity_scores: Lista di similarity scores ordinati (decrescente)
            
        Returns:
            (confidence_score, confidence_category)
        """
        if not similarity_scores:
            return 0.0, "none"
        
        # Strategia 1: Weighted average con decay esponenziale
        weights = [0.5, 0.3, 0.15, 0.05]  # Primi 4 risultati pesano di più
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i, score in enumerate(similarity_scores[:4]):  # Usa solo top 4
            weight = weights[i] if i < len(weights) else 0.0
            weighted_sum += score * weight
            total_weight += weight
        
        confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Penalizza se il top result è troppo basso
        top_score = similarity_scores[0]
        if top_score < 0.5:
            confidence *= 0.8  # Riduci del 20%
        
        # Determina categoria
        if confidence >= self.confidence_threshold_high:
            category = "high"
        elif confidence >= self.confidence_threshold_low:
            category = "medium"
        else:
            category = "low"
        
        return confidence, category
    
    def assemble_context(
        self,
        chunks: List[Dict[str, Any]],
        max_chunks: int = None
    ) -> str:
        """
        Assembla chunks in un context unico per LLM
        
        Args:
            chunks: Lista di chunks con text e metadata
            max_chunks: Numero massimo di chunks da usare
            
        Returns:
            Context formattato con separatori e citazioni
        """
        if max_chunks is None:
            max_chunks = self.context_max_chunks
        
        context_parts = []
        
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            # Formatta ogni chunk con metadata
            source = chunk['metadata']['source']
            page = chunk['metadata']['page']
            text = chunk['text']
            similarity = chunk.get('similarity_score', 0.0)
            
            chunk_context = (
                f"[Source {i}: {source}, Page {page}, Relevance: {similarity:.2f}]\n"
                f"{text}\n"
            )
            context_parts.append(chunk_context)
        
        # Unisci con separatori chiari
        assembled = "\n" + "="*60 + "\n".join(context_parts)
        
        return assembled
    
    def should_fallback_to_web(
        self,
        confidence: float,
        confidence_category: str,
        top_similarity: float
    ) -> bool:
        """
        Decide se usare web search fallback
        
        Regole:
        1. Low confidence (< 0.4) → sempre web
        2. Medium confidence (0.4-0.6) → web se top_similarity < 0.55
        3. High confidence (>= 0.6) → mai web (usa PDF)
        
        Args:
            confidence: Confidence aggregato
            confidence_category: "high", "medium", "low"
            top_similarity: Similarity del miglior chunk
            
        Returns:
            True se deve usare web search
        """
        if confidence_category == "low":
            return True
        
        if confidence_category == "medium":
            # Medium confidence ma top result molto basso → usa web
            if top_similarity < 0.55:
                return True
            return False
        
        # High confidence → usa sempre PDF
        return False
    
    def query(
        self,
        question: str,
        n_results: int = None,
        verbose: bool = True
    ) -> QueryResult:
        """
        Esegue query completa con confidence scoring
        
        Args:
            question: Domanda dell'utente
            n_results: Override default n_results
            verbose: Stampa info di debug
            
        Returns:
            QueryResult con tutti i dati necessari
        """
        if n_results is None:
            n_results = self.n_results
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 QUERY: {question}")
            print(f"{'='*60}")
        
        # Step 1: Search
        raw_results = self.embedding_system.search(
            query=question,
            n_results=n_results
        )
        
        # Step 2: Format results
        formatted_chunks = self.embedding_system.format_search_results(raw_results)
        
        if not formatted_chunks:
            if verbose:
                print("❌ Nessun risultato trovato")
            
            return QueryResult(
                query=question,
                confidence=0.0,
                should_use_web_fallback=True,
                retrieved_chunks=[],
                assembled_context="",
                confidence_category="none",
                top_similarity=0.0,
                avg_similarity=0.0
            )
        
        # Step 3: Extract similarity scores
        similarity_scores = [chunk['similarity_score'] for chunk in formatted_chunks]
        top_similarity = similarity_scores[0]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Step 4: Calculate aggregate confidence
        confidence, confidence_category = self.calculate_aggregate_confidence(
            similarity_scores
        )
        
        # Step 5: Decision logic
        use_web_fallback = self.should_fallback_to_web(
            confidence, confidence_category, top_similarity
        )
        
        # Step 6: Assemble context
        assembled_context = self.assemble_context(formatted_chunks)
        
        # Step 7: Print report (se verbose)
        if verbose:
            self._print_query_report(
                formatted_chunks[:3],  # Mostra solo top 3
                confidence,
                confidence_category,
                use_web_fallback,
                top_similarity,
                avg_similarity
            )
        
        return QueryResult(
            query=question,
            confidence=confidence,
            should_use_web_fallback=use_web_fallback,
            retrieved_chunks=formatted_chunks,
            assembled_context=assembled_context,
            confidence_category=confidence_category,
            top_similarity=top_similarity,
            avg_similarity=avg_similarity
        )
    
    def _print_query_report(
        self,
        top_chunks: List[Dict[str, Any]],
        confidence: float,
        confidence_category: str,
        use_web_fallback: bool,
        top_similarity: float,
        avg_similarity: float
    ):
        """Stampa report dettagliato della query"""
        
        # Confidence summary
        print(f"\n📊 CONFIDENCE ANALYSIS:")
        print(f"   Aggregate Confidence: {confidence:.3f}")
        print(f"   Category: {confidence_category.upper()}")
        print(f"   Top Similarity: {top_similarity:.3f}")
        print(f"   Avg Similarity: {avg_similarity:.3f}")
        
        # Decision
        print(f"\n🎯 DECISION:")
        if use_web_fallback:
            print(f"   ⚠️  USE WEB SEARCH FALLBACK")
            print(f"   Reason: Confidence too low or no relevant PDF content")
        else:
            print(f"   ✅ USE PDF CONTENT")
            print(f"   Reason: High confidence in retrieved chunks")
        
        # Top results
        print(f"\n📚 TOP RETRIEVED CHUNKS:")
        for chunk in top_chunks:
            print(f"\n   [{chunk['rank']}] Similarity: {chunk['similarity_score']:.3f}")
            print(f"   Source: {chunk['metadata']['source']} (Page {chunk['metadata']['page']})")
            print(f"   Text: {chunk['text'][:120]}...")
        
        print(f"\n{'='*60}\n")