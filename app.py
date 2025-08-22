import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import time
from datetime import datetime

# LangChain v0.2+ compatible imports
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Custom UI templates
from htmlTemplates import css, bot_template, user_template


# ========== Enhanced PDF Processing Functions ==========
def get_pdf_text_with_metadata(pdf_docs):
    """Extract text from PDFs while maintaining source information"""
    documents = []
  
    
    st.info("📊 Starting PDF text extraction...")
    
    for i, pdf in enumerate(pdf_docs):
        st.write(f"📄 Processing: **{pdf.name}**")
        
        try:
            reader = PdfReader(pdf)
            pdf_text = ""
            page_count = len(reader.pages)
            
            # Progress bar for each PDF
            progress = st.progress(0)
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                pdf_text += page_text
                progress.progress((page_num + 1) / page_count)
            
            progress.empty()
            
            documents.append({
                'filename': pdf.name,
                'text': pdf_text,
                'pages': page_count,
                'characters': len(pdf_text)
            })
            
            st.success(f"✅ Extracted {len(pdf_text)} characters from {page_count} pages")
            
        except Exception as e:
            st.error(f"❌ Error processing {pdf.name}: {str(e)}")
    
    return documents


def get_text_chunks_with_metadata(documents):
    """Split text into chunks while preserving source information"""
    st.info("✂️ Starting text chunking process...")
    
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    all_chunks = []
    chunk_metadata = []
    
    for doc in documents:
        st.write(f"📝 Chunking: **{doc['filename']}**")
        
        chunks = splitter.split_text(doc['text'])
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                'source': doc['filename'],
                'chunk_id': i,
                'chunk_size': len(chunk)
            })
        
        st.success(f"✅ Created {len(chunks)} chunks from {doc['filename']}")
    
    # Display chunking summary
    st.markdown("### 📋 Chunking Summary")
    summary_data = {}
    for metadata in chunk_metadata:
        source = metadata['source']
        if source not in summary_data:
            summary_data[source] = {'count': 0, 'total_size': 0}
        summary_data[source]['count'] += 1
        summary_data[source]['total_size'] += metadata['chunk_size']
    
    for source, data in summary_data.items():
        st.write(f"📄 **{source}**: {data['count']} chunks, {data['total_size']:,} characters")
    
    return all_chunks, chunk_metadata


# ========== Enhanced Vector Store ==========
def get_vectorstore_with_metadata(text_chunks, chunk_metadata):
    """Create vector store with metadata for source attribution"""
    st.info("🧠 Creating vector embeddings...")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS store with metadata
    metadatas = [{'source': meta['source'], 'chunk_id': meta['chunk_id']} for meta in chunk_metadata]
    
    vectorstore = FAISS.from_texts(
        texts=text_chunks, 
        embedding=embeddings,
        metadatas=metadatas
    )
    
    st.success(f"✅ Created vector store with {len(text_chunks)} embeddings")
    
    return vectorstore
# ========== Build Conversation Chain Dynamically ==========
def build_conversation_chain(vectorstore, response_type):
    """Build a conversation chain using the current response type"""
    if response_type == "Concise":
        template = """
You are a helpful AI assistant. Answer the user's question **briefly and concisely in 2-3 sentences**.
Use only the context provided.
Context: {context}
Question: {question}
Answer:"""
    else:
        template = """
You are a helpful AI assistant. Answer the user's question **in a detailed, well-explained manner**.
Provide examples, explanations, and reasoning when possible.
Use only the context provided.
Context: {context}
Question: {question}
Answer:"""

    qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    llm = ChatGroq(model="LLaMA3-8b-8192", temperature=0.7)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )






def handle_userinput_with_sources(user_question):
    """Handle user input and display sources"""
    with st.spinner("🤔 Thinking and searching through your documents..."):
        # Rebuild chain per question to respect current response type

        response = st.session_state.conversation({'question': user_question})

        if "chat_history" not in st.session_state or st.session_state.chat_history is None:
            st.session_state.chat_history = []

        st.session_state.chat_history.extend(response['chat_history'])


        # Get source documents and answer
        source_docs = response.get('source_documents', [])
        answer = response.get('answer', '')

        # Display the conversation
        chat_container = st.container()
        with chat_container:
            # User message
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_question)

            # Bot message with sources
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)
                st.caption(f"✍️ Response Style: {st.session_state.get('response_type', 'Concise')}")

                # Display sources if available
                if source_docs:
                    st.markdown("---")
                    st.markdown("### 📚 **Sources Used:**")

                    # Group sources by filename
                    sources_by_file = {}
                    for doc in source_docs:
                        source = doc.metadata.get('source', 'Unknown')
                        if source not in sources_by_file:
                            sources_by_file[source] = []
                        sources_by_file[source].append(doc)

                    # Display sources in expandable sections
                    for filename, docs in sources_by_file.items():
                        with st.expander(f"📄 {filename} ({len(docs)} chunks used)", expanded=False):
                            for i, doc in enumerate(docs, 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                                if i < len(docs):
                                    st.markdown("---")


def display_full_chat_history():
    """Display the complete chat history with sources"""
    if st.session_state.chat_history:
        st.markdown("### 📋 Complete Conversation History")
        
        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                # User message
                with st.chat_message("user", avatar="👤"):
                    st.markdown(st.session_state.chat_history[i].content)
                
                # Bot message
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(st.session_state.chat_history[i + 1].content)


# ========== Enhanced Streamlit App ==========
def main():
    load_dotenv()
    
    # Enhanced page configuration
    st.set_page_config(
        page_title="PDF Chat Assistant with Sources", 
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.write(css, unsafe_allow_html=True)
    
    # Enhanced modern styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .source-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .processing-log {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        font-family: monospace;
        font-size: 0.9em;
    }
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .stats-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .stats-label {
        font-size: 0.9rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

    # Session state setup
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processing_logs" not in st.session_state:
        st.session_state.processing_logs = []
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = {}
    if "last_question" not in st.session_state:
        st.session_state.last_question = None


    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>📚 PDF Chat Assistant with Source Attribution</h1>
        <p>Upload PDFs, ask questions, and see exactly which documents provided each answer</p>
    </div>
    """, unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface section
        st.markdown("### 💬 Chat Interface")
        
        # Status indicator with stats
        if st.session_state.conversation and st.session_state.document_stats:
            stats = st.session_state.document_stats
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{stats.get('total_files', 0)}</div>
                    <div class="stats-label">PDFs Loaded</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stat2:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{stats.get('total_chunks', 0)}</div>
                    <div class="stats-label">Text Chunks</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stat3:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{stats.get('total_pages', 0)}</div>
                    <div class="stats-label">Total Pages</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.success("✅ Ready to chat! Documents processed and indexed.")
        else:
            st.info("ℹ️ Upload and process your PDFs to start chatting with source attribution.")
        
        # Question input
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the key findings mentioned in the research papers?",
            help="Ask specific questions to get detailed answers with source references"
        )
        
        # Handle user input with dynamic conversation chain rebuild
        if user_question:
            st.session_state.last_question = user_question  # Save current question

            current_type = st.session_state.get("response_type", "Concise")
            
            # Rebuild conversation chain if it doesn't exist or if response type changed
            if ("conversation" not in st.session_state
                or st.session_state.conversation is None
                or st.session_state.get("conversation_type") != current_type):
                
                if "vectorstore" in st.session_state:
                    st.session_state.conversation = build_conversation_chain(
                        st.session_state.vectorstore,
                        current_type
                    )
                    st.session_state.conversation_type = current_type
                else:
                    st.warning("⚠️ Please upload and process PDFs first using the sidebar.")
            
            # If conversation chain exists, handle user input
            if "conversation" in st.session_state and st.session_state.conversation:
                handle_userinput_with_sources(user_question)
                
        elif user_question:
            st.warning("⚠️ Please upload and process PDFs first using the sidebar.")
        

    
    with col2:
        # Enhanced info panel
        st.markdown("### ℹ️ How it works")
        
        with st.expander("📖 Instructions", expanded=True):
            st.markdown("""
            1. **Upload PDFs** 📄 - Use sidebar to upload multiple PDFs
            2. **Process Documents** ⚙️ - Watch the detailed processing logs  
            3. **Ask Questions** ❓ - Get answers with source attribution
            4. **View Sources** 🔍 - See exactly which PDFs and chunks were used
            """)
        
        with st.expander("🆕 New Features"):
            st.markdown("""
            - **Source Attribution** - See which PDFs answered your question
            - **Chunk Visualization** - View the exact text chunks used
            - **Processing Logs** - Detailed breakdown of document processing
            - **Document Statistics** - Complete overview of your knowledge base
            """)

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("## 📁 Document Management")
        st.divider()
    
        # Upload PDFs
        st.markdown("### 📤 Upload Documents")
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload multiple PDFs for comprehensive question answering"
        )
    
        if pdf_docs:
            st.markdown("### 📋 Uploaded Files")
            for i, pdf in enumerate(pdf_docs, 1):
                st.markdown(f"**{i}.** {pdf.name}")
                st.caption(f"Size: {pdf.size / 1024:.1f} KB")
            st.divider()
    
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if pdf_docs:
                start_time = time.time()
                with st.expander("📊 Processing Logs", expanded=True):
                    # Extract text with metadata
                    documents = get_pdf_text_with_metadata(pdf_docs)
                    # Create chunks with metadata
                    text_chunks, chunk_metadata = get_text_chunks_with_metadata(documents)
                    # Create vector store
                    vectorstore = get_vectorstore_with_metadata(text_chunks, chunk_metadata)
                    st.session_state.vectorstore = vectorstore
                    # Build conversation chain right after vectorstore creation
                    st.session_state.conversation = build_conversation_chain(
                        vectorstore,
                        st.session_state.get("response_type", "Concise")
                    )
                    st.info("⚡ Vector store saved. Ready to answer questions with current mode.")
                    # Store statistics
                    st.session_state.document_stats = {
                        'total_files': len(documents),
                        'total_chunks': len(text_chunks),
                        'total_pages': sum(doc['pages'] for doc in documents),
                        'total_characters': sum(doc['characters'] for doc in documents),
                        'processing_time': round(time.time() - start_time, 2)
                    }
                    st.success(f"✅ Processing completed in {st.session_state.document_stats['processing_time']} seconds!")
                st.balloons()
            else:
                st.error("❌ Please upload at least one PDF file.")
    
        st.divider()
    
        # Response Style Selection
        st.markdown("### 📝 Response Style")
        response_type = st.radio(
            "Choose response style:",
            ("Concise", "Detailed"),
            horizontal=True
        )
    
        previous_type = st.session_state.get("conversation_type")
        st.session_state.response_type = response_type
    
        if previous_type and previous_type != response_type:
            st.session_state.conversation = build_conversation_chain(
                st.session_state.vectorstore,
                response_type
            )
            st.session_state.conversation_type = response_type
            st.session_state.rerun_last_question = True

    
    # --- Outside sidebar, in main chat column ---
    if st.session_state.get("rerun_last_question") and st.session_state.last_question:
        handle_userinput_with_sources(st.session_state.last_question)
        st.session_state.rerun_last_question = False  # reset flag



        

        
        # Show document statistics
        if st.session_state.document_stats:
            st.divider()
            st.markdown("### 📊 Knowledge Base Stats")
            stats = st.session_state.document_stats
            
            st.metric("Documents", stats['total_files'])
            st.metric("Text Chunks", stats['total_chunks'])
            st.metric("Total Pages", stats['total_pages'])
            st.metric("Processing Time", f"{stats['processing_time']}s")
        
        st.divider()
        
        # Technical details
        with st.expander("🛠️ Technical Details"):
            st.markdown("""
            **AI Model:** LLaMA3-8b-8192  
            **Embeddings:** all-MiniLM-L6-v2 (Sentence Transformers)  
            **Vector Store:** FAISS with metadata  
            **Chunk Size:** 1000 characters  
            **Chunk Overlap:** 200 characters  
            **Retrieval:** Top 4 relevant chunks  
            """)
        
        with st.expander("💡 Pro Tips"):
            st.markdown("""
            - **Source Tracking:** Each answer shows which PDFs were used
            - **Chunk Preview:** Click source expandables to see exact text
            - **Multiple Sources:** Answers may combine info from several PDFs
            - **Context Awareness:** Follow-up questions maintain conversation context
            """)


if __name__ == '__main__':
    main()
