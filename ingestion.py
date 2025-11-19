"""
PDF Ingestion System for RAG
Handles PDF loading, cleaning, and chunking with metadata
"""

from typing import List, Dict, Any
from pathlib import Path
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFIngestion:
    """Gestisce l'ingestion di PDF con chunking intelligente"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """
        Args:
            chunk_size: Dimensione target dei chunks (in caratteri)
            chunk_overlap: Sovrapposizione tra chunks per continuità
            min_chunk_size: Dimensione minima chunk (scarta chunks più piccoli)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Text splitter con separatori gerarchici
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Separazioni grandi (capitoli)
                "\n\n",    # Paragrafi
                "\n",      # Righe
                ". ",      # Frasi
                " ",       # Parole
                ""         # Caratteri (ultimo resort)
            ]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Estrae testo da PDF con metadata per pagina
        
        Args:
            pdf_path: Path al file PDF
            
        Returns:
            Lista di dict con testo e metadata per ogni pagina
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF non trovato: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File non è un PDF: {pdf_path}")
        
        reader = PdfReader(str(pdf_path))
        pages_data = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            
            if text.strip():  # Salta pagine vuote
                pages_data.append({
                    'text': text,
                    'page_number': page_num,
                    'source': pdf_path.name,
                    'total_pages': len(reader.pages)
                })
        
        print(f"✓ Estratte {len(pages_data)} pagine da {pdf_path.name}")
        return pages_data
    
    def clean_text(self, text: str) -> str:
        """
        Pulisce il testo da artefatti comuni dei PDF
        
        Args:
            text: Testo grezzo dal PDF
            
        Returns:
            Testo pulito
        """
        # Rimuovi multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Rimuovi spazi multipli
        text = re.sub(r' {2,}', ' ', text)
        
        # Rimuovi trattini di interruzione parola (word breaks)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Rimuovi header/footer patterns comuni (numeri pagina isolati)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        return text.strip()
    
    def create_chunks(
        self, 
        pages_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Crea chunks dal testo delle pagine con metadata enriched
        
        Args:
            pages_data: Lista di dict con testo e metadata delle pagine
            
        Returns:
            Lista di chunks con metadata completi
        """
        all_chunks = []
        
        for page_data in pages_data:
            # Pulisci il testo
            clean_text = self.clean_text(page_data['text'])
            
            # Crea chunks dal testo della pagina
            text_chunks = self.text_splitter.split_text(clean_text)
            
            # Aggiungi metadata a ogni chunk
            for idx, chunk_text in enumerate(text_chunks):
                # Salta chunks troppo piccoli (spesso artefatti)
                if len(chunk_text) < self.min_chunk_size:
                    continue
                
                chunk = {
                    'text': chunk_text,
                    'metadata': {
                        'source': page_data['source'],
                        'page': page_data['page_number'],
                        'total_pages': page_data['total_pages'],
                        'chunk_index': idx,
                        'char_count': len(chunk_text)
                    }
                }
                all_chunks.append(chunk)
        
        return all_chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Pipeline completo: estrazione → pulizia → chunking
        
        Args:
            pdf_path: Path al file PDF
            
        Returns:
            Lista di chunks pronti per l'embedding
        """
        print(f"\n📄 Processing: {pdf_path}")
        
        # Step 1: Estrai testo
        pages_data = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Crea chunks
        chunks = self.create_chunks(pages_data)
        
        print(f"✓ Creati {len(chunks)} chunks")
        print(f"  Avg size: {sum(c['metadata']['char_count'] for c in chunks) // len(chunks)} chars")
        
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Processa tutti i PDF in una directory
        
        Args:
            directory_path: Path alla directory con PDF
            
        Returns:
            Lista combinata di tutti i chunks
        """
        dir_path = Path(directory_path)
        pdf_files = list(dir_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  Nessun PDF trovato in {directory_path}")
            return []
        
        print(f"\n📁 Trovati {len(pdf_files)} PDF")
        
        all_chunks = []
        for pdf_file in pdf_files:
            try:
                chunks = self.process_pdf(str(pdf_file))
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"❌ Errore con {pdf_file.name}: {e}")
        
        print(f"\n✅ Totale: {len(all_chunks)} chunks da {len(pdf_files)} PDF")
        return all_chunks