"""
Streamlit UI per Sistema RAG
Interface grafica completa con chat, upload PDF, settings
"""

import streamlit as st
from pathlib import Path
import time
from typing import Optional

# Import dei moduli del sistema
from main import RAGSystem

# Page config
st.set_page_config(
    page_title="RAG System - Q&A on Documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Inizializza variabili di sessione"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'total_docs' not in st.session_state:
        st.session_state.total_docs = 0


def initialize_rag_system(tavily_key: Optional[str], enable_web: bool):
    """Inizializza il sistema RAG"""
    with st.spinner("🚀 Initializing RAG System..."):
        try:
            rag = RAGSystem(
                persist_directory="./chroma_db",
                collection_name="pdf_knowledge_base",
                llm_model="mistral:7b",
                tavily_api_key=tavily_key if enable_web else None,
                enable_web_search=enable_web
            )
            
            st.session_state.rag_system = rag
            st.session_state.total_docs = rag.embedding_system.collection.count()
            
            # Auto-carica PDF tutorial se DB vuoto
            if st.session_state.total_docs == 0:
                default_pdf = Path("./data/SimpleAgent- Pytutorial.pdf")
                if default_pdf.exists():
                    rag.ingest_pdf(str(default_pdf))
                    st.session_state.total_docs = rag.embedding_system.collection.count()
                    st.success(f"✅ RAG System ready! Pre-loaded {st.session_state.total_docs} chunks from tutorial.")
                else:
                    st.success("✅ RAG System ready! (No default PDF found)")
            else:
                st.success(f"✅ RAG System ready! {st.session_state.total_docs} documents in KB.")
            
            return True
        except Exception as e:
            st.error(f"❌ Error initializing system: {e}")
            return False


def display_chat_message(role: str, content: str, metadata: dict = None):
    """Mostra messaggio chat con styling"""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {role.title()}:</strong><br>
        {content}
    </div>
    """, unsafe_allow_html=True)
    
    # Mostra metadata se disponibili
    if metadata and role == "assistant":
        with st.expander("📊 Response Details"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                confidence = metadata.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.2%}")
                
                # Progress bar colorata
                if confidence >= 0.7:
                    color = "green"
                elif confidence >= 0.4:
                    color = "orange"
                else:
                    color = "red"
                st.progress(confidence)
            
            with col2:
                source_type = metadata.get('source_type', 'unknown').upper()
                st.metric("Source", source_type)
                
                if metadata.get('used_web_fallback'):
                    st.info("🌐 Web fallback used")
            
            with col3:
                response_time = metadata.get('response_time', 0)
                st.metric("Response Time", f"{response_time:.2f}s")
            
            # Sources
            sources = metadata.get('sources', [])
            if sources:
                st.markdown("**📚 Sources:**")
                for i, source in enumerate(sources[:3], 1):
                    if source_type == "PDF":
                        st.markdown(f"""
                        <div class="source-box">
                            [{i}] {source['source']} - Page {source['page']} 
                            (Relevance: {source['similarity']:.2f})
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # WEB
                        st.markdown(f"""
                        <div class="source-box">
                            [{i}] {source['title']}<br>
                            <a href="{source['url']}" target="_blank">{source['url']}</a>
                            (Score: {source['score']:.2f})
                        </div>
                        """, unsafe_allow_html=True)


def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">📚 RAG System - Document Q&A</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Web Search Configuration
        st.subheader("🌐 Web Search")
        enable_web_search = st.checkbox("Enable web fallback", value=False)
        
        tavily_key = None
        if enable_web_search:
            tavily_key = st.text_input(
                "Tavily API Key",
                type="password",
                help="Enter your Tavily API key for web search"
            )
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            initialize_rag_system(tavily_key, enable_web_search)
        
        st.divider()
        
        # PDF Upload
        st.subheader("📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=['pdf'],
            help="Upload PDF to add to knowledge base"
        )
        
        if uploaded_file and st.session_state.rag_system:
            if st.button("📥 Process PDF"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = f"./temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process PDF
                        st.session_state.rag_system.ingest_pdf(temp_path)
                        
                        # Update count
                        st.session_state.total_docs = st.session_state.rag_system.embedding_system.collection.count()
                        
                        # Cleanup
                        Path(temp_path).unlink()
                        
                        st.success(f"✅ PDF processed! Total docs: {st.session_state.total_docs}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        st.divider()
        
        # System Stats
        if st.session_state.rag_system:
            st.subheader("📊 System Stats")
            st.metric("Documents in KB", st.session_state.total_docs)
            st.metric("Chat Messages", len(st.session_state.chat_history))
        
        st.divider()
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main area
    if not st.session_state.rag_system:
        # Welcome screen
        st.info("""
        👋 **Welcome to RAG System!**
        
        This is a Retrieval-Augmented Generation system that can answer questions about your documents.
        
        **Features:**
        - 📄 Upload and query PDF documents
        - 🧠 Intelligent confidence-based routing
        - 🌐 Automatic web fallback for missing info
        - 🔍 Source citations and transparency
        
        **Getting Started:**
        1. Configure settings in the sidebar
        2. Click "Initialize System"
        3. Upload a PDF (optional)
        4. Start asking questions!
        """)
        
        st.warning("⚠️ Please initialize the system using the sidebar settings.")
    
    else:
        # Chat interface
        st.subheader("💬 Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(
                role=message['role'],
                content=message['content'],
                metadata=message.get('metadata')
            )
        
        # Chat input
        user_input = st.chat_input("Ask a question about your documents...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Display user message
            display_chat_message('user', user_input)
            
            # Get response
            with st.spinner("🤖 Thinking..."):
                try:
                    response = st.session_state.rag_system.query(
                        user_input,
                        verbose=False
                    )
                    
                    # Prepare metadata
                    metadata = {
                        'confidence': response.confidence,
                        'source_type': response.source_type,
                        'response_time': response.response_time,
                        'used_web_fallback': response.used_web_fallback,
                        'sources': response.sources
                    }
                    
                    # Add assistant message
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response.answer,
                        'metadata': metadata
                    })
                    
                    # Display assistant message
                    display_chat_message('assistant', response.answer, metadata)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            
            # Rerun to update chat
            st.rerun()


if __name__ == "__main__":
    main()