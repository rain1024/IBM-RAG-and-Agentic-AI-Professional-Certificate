import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import uuid

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="LangGraph Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "workflow_app" not in st.session_state:
    # Initialize the workflow app
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.7,
        max_tokens=1000
    )
    
    workflow = StateGraph(state_schema=MessagesState)
    
    def call_model(state: MessagesState):
        system_prompt = (
            "You are a helpful assistant. "
            "Answer all questions to the best of your ability."
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": response}
    
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    
    memory = MemorySaver()
    st.session_state.workflow_app = workflow.compile(checkpointer=memory)

# App title and description
st.title("🤖 LangGraph Chat Assistant")
st.markdown("Chat with your AI assistant powered by LangGraph and Azure OpenAI")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Thread ID management
    st.subheader("Conversation Thread")
    st.write(f"**Current Thread ID:** `{st.session_state.thread_id[:8]}...`")
    
    if st.button("🔄 New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    # Model parameters
    st.subheader("Model Settings")
    st.write("**Temperature:** 0.7")
    st.write("**Max Tokens:** 1000")
    
    # State Messages section
    st.subheader("📋 State Messages")
    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        current_state = st.session_state.workflow_app.get_state(config)
        
        if current_state and current_state.values.get("messages"):
            state_messages = current_state.values["messages"]
            st.write(f"**Total Messages:** {len(state_messages)}")
            
            with st.expander("View State Messages", expanded=False):
                for i, msg in enumerate(state_messages):
                    if hasattr(msg, 'type'):
                        msg_type = msg.type
                    else:
                        msg_type = type(msg).__name__
                    
                    if hasattr(msg, 'content'):
                        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        st.write(f"**{i+1}. {msg_type}:** {content}")
                    else:
                        st.write(f"**{i+1}. {msg_type}**")
        else:
            st.write("No messages in state yet")
    except Exception as e:
        st.write(f"Error reading state: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response from the workflow
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Invoke the workflow
                response = st.session_state.workflow_app.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config
                )
                
                # Extract the assistant's response
                assistant_response = response['messages'][-1].content
                
                # Display the response
                st.markdown(assistant_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
                # Rerun to update the state messages section in sidebar
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.write("Please check your Azure OpenAI configuration and try again.")

# Usage 
# streamlit run script06_2.py