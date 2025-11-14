"""
Hallucination Detector - Sistema di rilevamento allucinazioni LLM
Usa Natural Language Inference (NLI) per verificare fedeltà alle fonti
"""

import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class HallucinationScore:
    """Risultato dell'analisi di allucinazione"""
    score: float  # 0-1, più alto = più fedele alle fonti
    category: str  # "high_accuracy", "medium_accuracy", "low_accuracy"
    traffic_light: str  # "🟢", "🟡", "🔴"
    confidence_level: str  # "HIGH", "MEDIUM", "LOW" (combinato con similarity)
    details: Dict[str, Any]  # Info aggiuntive per debug


class HallucinationDetector:
    """
    Rileva allucinazioni confrontando risposta LLM con context fornito
    Usa modello NLI (Natural Language Inference) per verificare entailment
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = None
    ):
        """
        Args:
            model_name: Modello NLI da usare
                - "cross-encoder/nli-deberta-v3-small" (VELOCE, ~140MB)
                - "cross-encoder/nli-deberta-v3-base" (migliore, ~400MB)
            device: "cuda" o "cpu" (auto-detect se None)
        """
        # Detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔍 Initializing Hallucination Detector...")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_name}")
        
        # Carica modello NLI
        self.model = CrossEncoder(model_name, device=self.device)
        
        print(f"   ✓ Hallucination Detector ready\n")
    
    def calculate_entailment_score(
        self,
        context: str,
        response: str
    ) -> float:
        """
        Calcola score di entailment (quanto response è supportata da context)
        
        Args:
            context: Context fornito all'LLM (chunks PDF o web results)
            response: Risposta generata dall'LLM
            
        Returns:
            Score 0-1 (più alto = più fedele)
        """
        # NLI model predice: contradiction, neutral, entailment
        # Ci interessa l'entailment score
        
        pairs = [(context, response)]
        scores = self.model.predict(pairs)
        
        # CrossEncoder ritorna logits [contradiction, neutral, entailment]
        # Prendiamo il softmax per avere probabilità
        import torch.nn.functional as F
        probs = F.softmax(torch.tensor(scores), dim=-1)
        
        # Entailment score (ultima dimensione)
        entailment_score = probs[2].item()
        
        return entailment_score
    
    def detect_hallucination(
        self,
        context: str,
        response: str,
        similarity_score: float = None,
        verbose: bool = False
    ) -> HallucinationScore:
        """
        Analisi completa di allucinazione con sistema a semaforo
        
        Args:
            context: Context fornito (chunks o web results)
            response: Risposta LLM
            similarity_score: Score similarity dalla query (opzionale)
            verbose: Stampa dettagli
            
        Returns:
            HallucinationScore con tutti i dati
        """
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 HALLUCINATION DETECTION")
            print(f"{'='*60}")
        
        # Calcola entailment score
        entailment = self.calculate_entailment_score(context, response)
        
        # Determina categoria accuracy
        if entailment >= 0.7:
            accuracy_category = "high_accuracy"
            accuracy_label = "HIGH"
        elif entailment >= 0.5:
            accuracy_category = "medium_accuracy"
            accuracy_label = "MEDIUM"
        else:
            accuracy_category = "low_accuracy"
            accuracy_label = "LOW"
        
        # Sistema semaforo combinato (similarity + hallucination)
        if similarity_score is not None:
            # Combinazione intelligente
            traffic_light, confidence = self._calculate_traffic_light(
                similarity_score, entailment
            )
        else:
            # Solo hallucination score
            if entailment >= 0.7:
                traffic_light = "🟢"
                confidence = "HIGH"
            elif entailment >= 0.5:
                traffic_light = "🟡"
                confidence = "MEDIUM"
            else:
                traffic_light = "🔴"
                confidence = "LOW"
        
        # Dettagli per debug
        details = {
            "entailment_score": entailment,
            "similarity_score": similarity_score,
            "context_length": len(context),
            "response_length": len(response)
        }
        
        result = HallucinationScore(
            score=entailment,
            category=accuracy_category,
            traffic_light=traffic_light,
            confidence_level=confidence,
            details=details
        )
        
        if verbose:
            self._print_analysis(result)
        
        return result
    
    def _calculate_traffic_light(
        self,
        similarity: float,
        hallucination: float
    ) -> Tuple[str, str]:
        """
        Calcola semaforo combinando similarity e hallucination
        
        Logica:
        - 🟢 Verde: Entrambi alti (>0.7)
        - 🟡 Giallo: Uno basso o entrambi medi
        - 🔴 Rosso: Entrambi bassi (<0.5)
        """
        
        # Entrambi alti
        if similarity >= 0.7 and hallucination >= 0.7:
            return "🟢", "HIGH"
        
        # Entrambi bassi
        if similarity < 0.5 and hallucination < 0.5:
            return "🔴", "LOW"
        
        # Almeno uno molto alto salva la situazione
        if similarity >= 0.8 or hallucination >= 0.8:
            return "🟢", "HIGH"
        
        # Almeno uno molto basso rovina tutto
        if similarity < 0.4 or hallucination < 0.4:
            return "🔴", "LOW"
        
        # Tutti gli altri casi: medio
        return "🟡", "MEDIUM"
    
    def _print_analysis(self, result: HallucinationScore):
        """Stampa analisi formattata"""
        
        print(f"\n📊 HALLUCINATION ANALYSIS:")
        print(f"   Accuracy Score: {result.score:.3f}")
        print(f"   Category: {result.category.replace('_', ' ').upper()}")
        
        if result.details.get('similarity_score'):
            sim = result.details['similarity_score']
            print(f"   Relevance Score: {sim:.3f}")
        
        print(f"\n🚦 QUALITY ASSESSMENT:")
        print(f"   {result.traffic_light} {result.confidence_level} CONFIDENCE")
        
        # Interpretazione
        if result.confidence_level == "HIGH":
            interpretation = "✅ Response is well-supported by sources"
        elif result.confidence_level == "MEDIUM":
            interpretation = "⚠️  Response partially supported, verify key claims"
        else:
            interpretation = "❌ Response may contain unsupported information"
        
        print(f"   {interpretation}")
        
        print(f"{'='*60}\n")
    
    def batch_detect(
        self,
        context_response_pairs: List[Tuple[str, str]],
        show_progress: bool = True
    ) -> List[HallucinationScore]:
        """
        Analisi batch per multiple risposte
        
        Args:
            context_response_pairs: Lista di (context, response) tuples
            show_progress: Mostra progress bar
            
        Returns:
            Lista di HallucinationScore
        """
        results = []
        
        total = len(context_response_pairs)
        
        for i, (context, response) in enumerate(context_response_pairs, 1):
            if show_progress:
                print(f"Processing {i}/{total}...", end='\r')
            
            result = self.detect_hallucination(
                context=context,
                response=response,
                verbose=False
            )
            results.append(result)
        
        if show_progress:
            print(f"✓ Processed {total} responses")
        
        return results


# ============= ESEMPIO D'USO =============

if __name__ == "__main__":
    print("="*70)
    print("STEP 6: HALLUCINATION DETECTOR TEST")
    print("="*70)
    
    # Inizializza detector
    detector = HallucinationDetector(
        model_name="cross-encoder/nli-deberta-v3-small",
        device=None  # Auto-detect
    )
    
    # Test Case 1: Risposta FEDELE (dovrebbe essere 🟢)
    print("\n" + "="*70)
    print("TEST 1: HIGH ACCURACY - Risposta fedele al context")
    print("="*70)
    
    context_1 = """
    Python lists are mutable sequences, typically used to store collections 
    of homogeneous items. Lists can be created using square brackets [].
    Example: my_list = [1, 2, 3, 4]
    """
    
    response_1 = """
    Python lists are mutable sequences that can store collections of items.
    They are created using square brackets, for example: [1, 2, 3, 4].
    """
    
    result_1 = detector.detect_hallucination(
        context=context_1,
        response=response_1,
        similarity_score=0.85,  # Simula similarity alta
        verbose=True
    )
    
    # Test Case 2: Risposta con ALLUCINAZIONI (dovrebbe essere 🔴 o 🟡)
    print("\n" + "="*70)
    print("TEST 2: LOW ACCURACY - Risposta con allucinazioni")
    print("="*70)
    
    context_2 = """
    Python lists are mutable sequences. You can modify, add, or remove elements.
    Lists preserve the order of elements.
    """
    
    response_2 = """
    Python lists are immutable sequences that cannot be changed after creation.
    Lists do not preserve order and elements are stored randomly.
    Lists can only contain strings and numbers.
    """
    
    result_2 = detector.detect_hallucination(
        context=context_2,
        response=response_2,
        similarity_score=0.65,
        verbose=True
    )
    
    # Test Case 3: Risposta MEDIA (dovrebbe essere 🟡)
    print("\n" + "="*70)
    print("TEST 3: MEDIUM ACCURACY - Risposta parzialmente supportata")
    print("="*70)
    
    context_3 = """
    Python dictionaries store key-value pairs. Keys must be immutable types.
    Example: my_dict = {"name": "Alice", "age": 30}
    """
    
    response_3 = """
    Python dictionaries store key-value pairs, where keys must be immutable.
    Dictionaries are the fastest data structure in Python for all use cases.
    They were invented by Guido van Rossum in 1995.
    """
    
    result_3 = detector.detect_hallucination(
        context=context_3,
        response=response_3,
        similarity_score=0.70,
        verbose=True
    )
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    tests = [
        ("Fedele", result_1),
        ("Allucinato", result_2),
        ("Parziale", result_3)
    ]
    
    for name, result in tests:
        print(f"\n{name}:")
        print(f"  {result.traffic_light} Score: {result.score:.3f}")
        print(f"  Confidence: {result.confidence_level}")
    
    print("\n" + "="*70)
    print("✅ HALLUCINATION DETECTOR TEST COMPLETATO!")
    print("="*70)
    print("\nIl sistema ora può:")
    print("  ✅ Rilevare allucinazioni con NLI")
    print("  ✅ Combinare similarity + accuracy")
    print("  ✅ Sistema semaforo 🟢🟡🔴")
    print("  ✅ Report dettagliati per ogni risposta")