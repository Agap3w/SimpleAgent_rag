## SimpleAgent ##
#### RAG agent per Q&A su pdf caricati, con fallback su websearch e UI minimal ####

### Elenco file ###

| File | Descrizione |
|------|-------------|
| ingestion.py | Estrae testo da PDF e lo divide in chunk |
| embeddings.py | Converte chunk in vettori numerici e li salva in ChromaDB |
| query_system.py | Cerca chunk simili e calcola confidence (decide: PDF o Web?) |
| llm_handler.py | Comunica con Mistral via Ollama |
| web_search_handler.py | Fallback Tavily quando confidence basso |
| rag_pipeline.py | Integra tutto in un flusso unico |
| main.py | main.py |
| streamlit_ui.py | UI |

\+ file/cartelle di progetto: pycache, chroma_db, data, venv_SimpleAgent, gitignore