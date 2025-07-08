import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from groq import Groq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from collections import deque
from vector import retriever, enhanced_retriever_instance
import re
import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import importlib.util
import requests
import streamlit as st


load_dotenv()

class BlandAI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            'authorization': self.api_key,
            'Content-Type': 'application/json'
        }
        
    def call(self, phone_number, pathway_id, first_sentence, **kwargs):
        url = 'https://api.bland.ai/v1/calls'
        data = {
            'phone_number': phone_number,
            'pathway_id': pathway_id,
            'first_sentence': first_sentence
        }
        # Add any additional keyword arguments to the payload
        data.update(kwargs)
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()
    
    def logs(self, call_id):
        url = 'https://api.bland.ai/v1/logs'
        data = {'call_id': call_id}
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()

    def hold(self, call_id):
        url = 'https://api.bland.ai/v1/hold'
        data = {'call_id': call_id}
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()

    def end_call(self, call_id):
        url = 'https://api.bland.ai/v1/end'
        data = {'call_id': call_id}
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()

# Initialize BlandAI with API key
bland_ai = BlandAI("org_cadc2d5640e34cedbced0d3627c9236efc7f1e0424737b6d039a53a8d622fe0bca47e1b19cd0e5e7a3f169")

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait() 


recognizer = sr.Recognizer()



def listen():
    with sr.Microphone() as source:
        print("Listening...")
        speak("I'm listening")

        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjusts for background noise

        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text

        except sr.WaitTimeoutError:
            print("I didn't hear anything, could you please repeat?")
            speak("I didn't hear anything, could you please repeat?")
            return ""

        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Could you please repeat?")
            speak("Sorry, I didn't catch that. Could you please repeat?")
            return ""

        except Exception as e:
            print(f"Error in speech recognition: {e}")
            speak("Sorry, there was an error in speech recognition. Please try again.")
            return ""



# Constants for memory management
MAX_MESSAGES = 10  # Maximum number of messages to keep in memory
SUMMARY_THRESHOLD = 8  # When to trigger summarization

client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
)

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional or logical response.",
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    memory: dict
    summary: str | None

def summarize_conversation(messages):
    """Summarize the conversation history to maintain context while reducing memory usage."""
    system_prompt = """Summarize the key points of this conversation in a concise way that preserves important context.\nFocus on main topics, decisions, and any important information shared."""

    def get_role(msg):
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        if isinstance(msg, HumanMessage):
            return "user"
        elif isinstance(msg, AIMessage):
            return "assistant"
        elif isinstance(msg, SystemMessage):
            return "system"
        else:
            return "unknown"

    formatted_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please summarize this conversation:\n" + "\n".join([f"{get_role(msg)}: {msg.content}" for msg in messages])}
    ]

    chat_completion = client.chat.completions.create(
        messages=formatted_messages,
        model="llama-3.3-70b-versatile"
    )

    return chat_completion.choices[0].message.content

def manage_memory(state: State) -> State:
    """Manage the conversation memory by implementing a sliding window and summarization."""
    messages = state["messages"]
    
    # If we have too many messages, summarize and keep only recent ones
    if len(messages) > MAX_MESSAGES:
        # Get messages to summarize (excluding the most recent ones)
        messages_to_summarize = messages[:-SUMMARY_THRESHOLD]
        recent_messages = messages[-SUMMARY_THRESHOLD:]
        
        # Create or update summary
        if state.get("summary"):
            summary_prompt = f"Previous summary: {state['summary']}\nNew messages to add to summary:"
            new_summary = summarize_conversation(messages_to_summarize)
            state["summary"] = new_summary
        else:
            state["summary"] = summarize_conversation(messages_to_summarize)
        
        # Keep only recent messages
        state["messages"] = recent_messages
    
    return state

def classify_message(state: State):
    last_message = state["messages"][-1]

    system_prompt = """Classify the user message into one of three categories:
    - 'call': if the user wants to place a call, make a call, or requests phone assistance
    - 'true': if it mentions any names or any confidential information relating to insurance policies in the query (but not a call request)
    - 'false': if it asks for basic doubts regarding their insurance policies without mentioning any of their confidential information in the query

    Examples of 'call' messages:
    - "I want to place a call"
    - "Can you call me?"
    - "Make a call to my number"
    - "I need to speak with someone on the phone"
    - "Call me at my number"
    - "Place a call to discuss my policy"

    Examples of 'true' messages:
    - "What is the premium amount for the policy number 1234567890?(see its mentioning the policy number)"
    - "My name is John and my policy number is 1234567890, can you tell me my premium amount?(see its mentioning the policy number and the name)"
    - mentioning of their names or any confidential information in the query.

    Examples of 'false' messages:
    - "What is the claim process for my policy?(see its asking for basic doubts regarding the policy)"
    - "How do I file a claim?(see its asking for basic doubts regarding the policy)"
    - "what is my policy number?(see its asking for basic doubts regarding the policy)"
    - no mentioning of their names or any confidential information in the query.
    
    Respond with only one word: either "call", "true", or "false".
    """

    formatted_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": last_message.content}
    ]

    chat_completion = client.chat.completions.create(
        messages=formatted_messages,
        model="llama-3.3-70b-versatile"
    )

    classification_response = chat_completion.choices[0].message.content.strip().lower()
    return {"message_type": classification_response}

def confid_checker(state: State):
    message_type = state.get("message_type", "false")
    if message_type == "call":
        return {"next": "call_placement"}
    elif message_type == "true":
        return {"next": "RAG_model"}
    else:
        return {"next": "FT_model"}


def RAG(state: State):
    state = manage_memory(state)
    last_message = state["messages"][-1]
    question = last_message.content
    print(f"Processing question: {question}")
    
    # Use enhanced retriever for family insurance queries
    details = enhanced_retriever_instance(question)
    
    clean_details = "\n".join([doc.page_content for doc in details])

    if clean_details:
        user_message = f"{question}\n\nHere are relevant insurance details:\n{clean_details}"
    else:
        user_message = question

    messages = [
        {
            "role": "system",
            "content": """You are an insurance agent who informs customers about their premium dues and answers insurance-related queries in a helpful tone.
            IMPORTANT RULES:
            1. Always maintain consistency in user information. If you've already identified a user's name and policy details, stick to those details throughout the conversation.
            2. If you're unsure about any user details, ask for clarification rather than making assumptions.
            3. When answering questions about user details, always check the conversation history and summary first.
            4. Never say you don't have information if it was previously shared in the conversation.
            5. Always include relevant policy details in your responses when they are known.
            6. Review the entire conversation history before responding.
            7. If the user introduces themselves with a new name, update their identity but maintain consistency with that new identity.
            8. When retrieving information, only use details that match the current user's identity.
            9. If retrieved information conflicts with the current user's identity, prioritize the user's stated identity.
            10. For family-related queries, group all policies by FamilyID and show all family members with their respective policies.
            11. When asked about family details, list all people under the same FamilyID with their policy information.
            12. Include nominee information when available in your responses.
            13. When responding to family insurance queries, organize the information clearly by family member, showing each person's policies separately.
            14. For family queries, provide a comprehensive overview of all family members' insurance coverage, including policy types, premiums, and expiry dates.
            15. Always mention the FamilyID when discussing family insurance to help users understand the grouping."""
        }
    ]

    # Add conversation history
    for msg in state["messages"]:
        if isinstance(msg, (HumanMessage, AIMessage)):
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            messages.append({"role": role, "content": msg.content})

    # Add summary if it exists
    if state.get("summary"):
        messages.append({
            "role": "system",
            "content": f"""Previous conversation summary: {state['summary']}
            Use this summary to maintain context and consistency in your responses."""
        })

    # Add the current question and retrieved details
    messages.append({
        "role": "user",
        "content": user_message
    })

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
    )

    result = chat_completion.choices[0].message.content
    final_result = re.sub(r"[*_#`]", "", result)

    return {
        "messages": [AIMessage(content=final_result)],
        "memory": state.get("memory", {}),
        "summary": state.get("summary")
    }


#Fine_tuned model
def FT(state: State, model, tokenizer):
    state = manage_memory(state)
    last_message = state["messages"][-1]

    # Compose the enhanced system prompt
    system_prompt = (
        "You are InsurAI, a professional and knowledgeable insurance assistant. You provide accurate, helpful, and comprehensive insurance information.\n\n"
        "CORE RESPONSIBILITIES:\n"
        "1. Provide detailed information about various insurance products and policies\n"
        "2. Explain insurance terms, conditions, and processes clearly\n"
        "3. Guide users through claim procedures and policy management\n"
        "4. Offer general insurance advice while maintaining professional boundaries\n"
        "5. Direct users to appropriate resources when specific personal advice is needed\n\n"
        "RESPONSE GUIDELINES:\n"
        "1. Be informative and educational - explain concepts clearly with specific details\n"
        "2. Maintain professional tone while being approachable and empathetic\n"
        "3. When discussing specific policies, provide comprehensive details including coverage, exclusions, and benefits\n"
        "4. Always mention important considerations like state availability, underwriting requirements, and waiting periods\n"
        "5. Encourage users to contact insurance companies or financial advisors for personalized advice\n"
        "6. Use clear structure with bullet points or numbered lists when appropriate\n"
        "7. Include relevant examples and scenarios to illustrate points\n"
        "8. Provide actionable next steps and recommendations\n"
        "9. Address user concerns proactively and offer solutions\n"
        "10. Use specific numbers, percentages, and timeframes when available\n\n"
        "DETAILED RESPONSE STRUCTURE:\n"
        "1. Start with a clear, direct answer to the user's question\n"
        "2. Provide comprehensive details with specific information\n"
        "3. Include relevant policy features, benefits, and limitations\n"
        "4. Mention important considerations and requirements\n"
        "5. Provide practical examples or scenarios\n"
        "6. Offer actionable recommendations and next steps\n"
        "7. End with a helpful summary or key takeaways\n\n"
        "IMPORTANT RULES:\n"
        "1. Always maintain consistency in user information throughout the conversation\n"
        "2. If unsure about user details, ask for clarification rather than making assumptions\n"
        "3. Check conversation history and summary before responding\n"
        "4. Never claim to lack information that was previously shared\n"
        "5. Include relevant policy details when known\n"
        "6. Review entire conversation history before responding\n"
        "7. Update user identity when new information is provided\n"
        "8. Prioritize user-stated identity over conflicting retrieved information\n"
        "9. For specific policy questions, provide detailed general information and recommend direct contact\n"
        "10. Always mention the importance of consulting with insurance professionals for personalized advice\n"
        "11. For family-related queries, group all policies by FamilyID and show all family members with their respective policies\n"
        "12. When asked about family details, list all people under the same FamilyID with their policy information\n"
        "13. Include nominee information when available in your responses\n"
        "14. When responding to family insurance queries, organize the information clearly by family member, showing each person's policies separately\n"
        "15. For family queries, provide a comprehensive overview of all family members' insurance coverage, including policy types, premiums, and expiry dates\n"
        "16. Always mention the FamilyID when discussing family insurance to help users understand the grouping\n"
        "17. Provide specific, actionable information rather than generic responses\n"
        "18. Include relevant statistics, examples, and comparisons when helpful\n"
        "19. Address potential concerns and objections proactively\n"
        "20. Offer multiple options or alternatives when applicable"
    )

    # Build the conversation prompt
    prompt = f"<|system|>\n{system_prompt}\n</s>\n"
    for msg in state["messages"]:
        if isinstance(msg, (HumanMessage, AIMessage)):
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            prompt += f"<|{role}|>\n{msg.content}\n</s>\n"
    if state.get("summary"):
        prompt += f"<|system|>\nPrevious conversation summary: {state['summary']}\nUse this summary to maintain context and consistency in your responses.\n</s>\n"
    prompt += f"<|system|>\nIMPORTANT: Provide a detailed, comprehensive response with specific information, examples, and actionable advice. Be thorough and helpful in your explanation.\n</s>\n"
    prompt += f"<|user|>\n{last_message.content}\n</s>\n<|assistant|>\n"

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.4,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.1,
            length_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response (after the last <|assistant|>)
    assistant_split = result.split("<|assistant|>")
    final_result = assistant_split[-1].strip() if len(assistant_split) > 1 else result.strip()
    final_result = re.sub(r"[*_#`]", "", final_result)

    return {
        "messages": [AIMessage(content=final_result)],
        "memory": state.get("memory", {}),
        "summary": state.get("summary")
    }


def call_placement(state: State):
    """Handle call placement requests using BlandAI"""
    state = manage_memory(state)
    last_message = state["messages"][-1]
    
    # Extract phone number from the message if provided
    phone_number = None
    message_content = last_message.content.lower()
    
    # Look for phone number patterns in the message
    import re
    phone_patterns = [
        r'\+?1?\s*\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}',  # US format
        r'\+?91\s*[0-9]{10}',  # Indian format
        r'\+?[0-9]{1,3}\s*[0-9]{10}',  # International format
        r'[0-9]{10}',  # Simple 10-digit
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, last_message.content)
        if match:
            phone_number = match.group(0)
            break
    
    # Extract payment amount if mentioned
    payment_amount = None
    payment_pattern = r'₹(\d+(?:,\d+)*(?:\.\d{2})?)'
    payment_match = re.search(payment_pattern, last_message.content)
    if payment_match:
        payment_amount = payment_match.group(1).replace(',', '')
    
    # Extract customer name if mentioned
    customer_name = None
    name_patterns = [
        r'calling (\w+(?:\s+\w+)*)',
        r'customer (\w+(?:\s+\w+)*)',
        r'(\w+(?:\s+\w+)*) regarding'
    ]
    for pattern in name_patterns:
        match = re.search(pattern, last_message.content, re.IGNORECASE)
        if match:
            customer_name = match.group(1)
            break
    
    # If no phone number found, ask for it
    if not phone_number:
        response = "I'd be happy to place a call for you! Could you please provide your phone number so I can connect you with our insurance specialist?"
        return {
            "messages": [AIMessage(content=response)],
            "memory": state.get("memory", {}),
            "summary": state.get("summary")
        }
    
    try:
        # Prepare call message based on context
        if payment_amount and customer_name:
            call_message = f"Hello! This is a call from InsurAI. We're calling {customer_name} regarding premium payment collection. The outstanding amount is ₹{payment_amount}. Please be ready to collect the payment and provide payment options."
        elif customer_name:
            call_message = f"Hello! This is a call from InsurAI. We're calling {customer_name} regarding their insurance policy. How can we assist you today?"
        else:
            call_message = "Hello! This is a call from InsurAI. We're here to assist you with your insurance needs."
        
        # Place the call using BlandAI
        call_response = bland_ai.call(
            phone_number=phone_number,
            pathway_id="67ca7361-290c-4582-9b5f-26afd460653e",
            first_sentence=call_message
        )
        
        if call_response.get("success"):
            call_id = call_response.get("call_id")
            response = f"Great! I've initiated a call to {phone_number}."
            
            if customer_name:
                response += f" Our AI specialist will be calling {customer_name}."
            
            if payment_amount:
                response += f" The call will focus on collecting the outstanding premium amount of ₹{payment_amount}."
            
            response += f" Your call ID is {call_id}. The AI agent will handle the conversation professionally and collect any required payments."
        else:
            error_msg = call_response.get("error", "Unknown error")
            response = f"I apologize, but I encountered an issue placing the call: {error_msg}. Please try again or contact our support team directly."
            
    except Exception as e:
        response = f"I apologize, but there was an error placing the call: {str(e)}. Please try again or contact our support team directly."
    
    return {
        "messages": [AIMessage(content=response)],
        "memory": state.get("memory", {}),
        "summary": state.get("summary")
    }


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", confid_checker)
graph_builder.add_node("RAG_model", RAG)
graph_builder.add_node("FT_model", FT)
graph_builder.add_node("call_placement", call_placement)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {
        "RAG_model": "RAG_model",
        "FT_model": "FT_model",
        "call_placement": "call_placement"
    }
)

graph_builder.add_edge("RAG_model", END)
graph_builder.add_edge("FT_model", END)
graph_builder.add_edge("call_placement", END)

graph = graph_builder.compile(checkpointer=MemorySaver())

def run_chatbot():
    config = {"configurable": {"thread_id": 1}}
    state = {"messages": [], "message_type": None, "memory": {}, "summary": None}

    print("Chatbot started. Say 'quit' to end the conversation.")
    speak("Hello! I'm your insurance assistant. How can I help you today?")
   

    while True:
        try:
            print("\n---------------------")
            user_input = input()

            if not user_input.strip():
                print("Empty input, skipping.")
                continue
            
            if user_input.lower().strip() == "quit":
                print("Goodbye! Have a great day!")
                break

            state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
            state = graph.invoke(state, config=config)

            if state.get("messages") and len(state["messages"]) > 0:
                last_message = state["messages"][-1]
                print(f"Assistant: {last_message.content}")
                speak(last_message.content)
            else:
                print("No valid response from model.")
                speak("Sorry, I didn't get a response.")

        except Exception as e:
            print(f"Error: {e}")
            speak("Sorry, there was an error. Please try again.")


if __name__ == "__main__":
    run_chatbot()

st.markdown("""
<div class="upload-section">
    <div class="upload-header">
        <h3 style="color:#00ff88;">🚀 AI Assistant Features</h3>
    </div>
    <p style="color:#f3f3f3;">The AI agent can help you with various insurance-related tasks!</p>
    <p style="color:#00d4ff;"><strong>📞 Call Placement:</strong></p>
    <ul style="color:#f3f3f3;">
        <li>I want to place a call</li>
        <li>Can you call me at +919744930824?</li>
        <li>Make a call to discuss my policy</li>
        <li>I need to speak with someone on the phone</li>
    </ul>
    <p style="color:#00d4ff;"><strong>👨‍👩‍👧‍👦 Family Insurance:</strong></p>
    <ul style="color:#f3f3f3;">
        <li>Show me family insurance details for John Smith</li>
        <li>What are the family policies for policy number 1234567890</li>
        <li>Get insurance details for my family members</li>
        <li>Show me all policies for my spouse and children</li>
    </ul>
</div>
""", unsafe_allow_html=True)