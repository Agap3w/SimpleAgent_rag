"""
Embeddings Generation & ChromaDB Storage System
Ottimizzato per GPU (RTX 3080)
"""

from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class EmbeddingSystem:
    """Gestisce embeddings con GPU e storage in ChromaDB"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "pdf_knowledge_base",
        persist_directory: str = "./chroma_db"
    ):
        """
        Args:
            model_name: Modello sentence-transformers
                - "all-MiniLM-L6-v2": Veloce, 384 dim (ok x basic)
                - "all-mpnet-base-v2": Migliore qualità, 768 dim
                - "multi-qa-mpnet-base-dot-v1": Ottimizzato per Q&A
            collection_name: Nome della collection ChromaDB
            persist_directory: Directory per persistenza DB
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Controlla disponibilità GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Device: {self.device}")
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU rilevata: {gpu_name}")
        
        # Carica modello embeddings
        print(f"📦 Caricamento modello: {model_name}...")
        self.model = SentenceTransformer(model_name, device=self.device) #con questa riga scarico il modello di embedding (91Mb) e lo carico sulla GPU
        print(f"✓ Modello caricato su {self.device}")
        
        # Inizializza ChromaDB
        self.client = chromadb.PersistentClient( 
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Ottieni o crea collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Usa cosine similarity per confrontare i vettori
        )
        
        print(f"✓ ChromaDB inizializzato: {persist_directory}")
        print(f"✓ Collection: {collection_name}")
        print(f"📊 Documenti esistenti: {self.collection.count()}\n")
    
    def generate_embeddings(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Genera embeddings con GPU acceleration
        
        Args:
            texts: Lista di testi da embeddare
            batch_size: Dimensione batch per GPU
            show_progress: Mostra progress bar
            
        Returns:
            Lista di embeddings (vettori)
        """
        print(f"🔢 Generazione embeddings per {len(texts)} chunks...")
        
        # Encode con GPU batching
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True  # Normalizza per cosine similarity
        )
        
        print(f"✓ Embeddings generati: shape {embeddings.shape}")
        return embeddings.tolist()
    
    def add_chunks_to_db(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> None:
        """
        Aggiunge chunks al database con embeddings
        
        Args:
            chunks: Lista di chunks dal PDF ingestion
            batch_size: Batch size per embeddings GPU
        """
        if not chunks:
            print("⚠️  Nessun chunk da aggiungere")
            return
        
        # Estrai testi e metadata
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Genera IDs univoci
        ids = [f"{chunk['metadata']['source']}_page{chunk['metadata']['page']}_chunk{chunk['metadata']['chunk_index']}" 
               for chunk in chunks]
        
        # Genera embeddings
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        
        # Aggiungi a ChromaDB
        print(f"💾 Salvataggio in ChromaDB...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings, 
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✅ {len(chunks)} chunks aggiunti al database")
        print(f"📊 Totale documenti: {self.collection.count()}")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Dict = None
    ) -> Dict[str, Any]:
        """
        Cerca chunks simili alla query
        
        Args:
            query: Domanda dell'utente
            n_results: Numero di risultati da ritornare
            where: Filtri metadata (es. {"source": "tutorial.pdf"})
            
        Returns:
            Risultati con documenti, metadati e scores
        """
        # Genera embedding della query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True
        ).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def format_search_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Formatta risultati search in formato leggibile
        
        Args:
            results: Output da self.search()
            
        Returns:
            Lista di dict con risultati formattati
        """
        formatted = []
        
        if not results['ids'] or not results['ids'][0]:
            return formatted
        
        for i in range(len(results['ids'][0])):
            formatted.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1 - results['distances'][0][i],  # Converti distance a similarity
                'rank': i + 1
            })
        
        return formatted
    
    def reset_database(self):
        """ATTENZIONE: Cancella tutto il database"""
        self.client.delete_collection(self.collection_name)
        print(f"🗑️  Database '{self.collection_name}' cancellato")
        
        # Ricrea collection vuota
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Collection ricreata vuota")