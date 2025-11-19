*** SimpleAgent ***
*** RAG agent per Q&A su pdf caricati, con fallback su websearch e UI minimal ***

** Elenco file **

1. ingestion.py          → Estrae testo da PDF e lo divide in chunk
2. embeddings.py         → Converte chunk in vettori numerici e li salva in ChromaDB
3. query_system.py       → Cerca chunk simili e calcola confidence (decide: PDF o Web?)
4. llm_handler.py        → Comunica con Mistral via Ollama
5. web_search_handler.py → Fallback Tavily quando confidence basso
6. rag_pipeline.py       → Integra tutto in un flusso unico
7. main.py               → main.py
8. streamlit_ui.py       → UI

00. pycache, chroma_db, data, venv_SimpleAgent, gitignore → file/cartelle di progetto