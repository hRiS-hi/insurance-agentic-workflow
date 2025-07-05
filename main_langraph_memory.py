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
from vector import retriever
import re
import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import importlib.util


load_dotenv()

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

    system_prompt = """Classify the user message as either:
    - 'true': if it mentions any names or any confidential information relating to insurance policies in the query.
    - 'false': if it asks for basic doubts regarding their insurance policies wihout mentioning any of their confidential information in the query.

    Examples of 'true' messages:
    - "What is the premium amount for the policy number 1234567890?(see its mentioning the policy number)"
    - "My name is John and my policy number is 1234567890, can you tell me my premium amount?(see its mentioning the policy number and the name)"
    - mentioning of their names or any confidential information in the query.

    Examples of 'false' messages:
    - "What is the claim process for my policy?(see its asking for basic doubts regarding the policy)"
    - "How do I file a claim?(see its asking for basic doubts regarding the policy)"
    - "what is my policy number?(see its asking for basic doubts regarding the policy)"
    - no mentioning of their names or any confidential information in the query.
    
    Respond with only one word: either "true" or "false".
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
    if message_type == "true":
        return {"next": "RAG_model"}

    return {"next": "FT_model"}


def RAG(state: State):
    state = manage_memory(state)
    last_message = state["messages"][-1]
    question = last_message.content
    print(f"Processing question: {question}")
    details = retriever.invoke(question)
    
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
            9. If retrieved information conflicts with the current user's identity, prioritize the user's stated identity."""
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
        "1. Be informative and educational - explain concepts clearly\n"
        "2. Maintain professional tone while being approachable\n"
        "3. When discussing specific policies (like Golden Plans), provide comprehensive details\n"
        "4. Always mention important considerations like state availability and underwriting requirements\n"
        "5. Encourage users to contact insurance companies or financial advisors for personalized advice\n"
        "6. Use clear structure with bullet points or numbered lists when appropriate\n"
        "7. Include relevant examples and scenarios to illustrate points\n\n"
        "IMPORTANT RULES:\n"
        "1. Always maintain consistency in user information throughout the conversation\n"
        "2. If unsure about user details, ask for clarification rather than making assumptions\n"
        "3. Check conversation history and summary before responding\n"
        "4. Never claim to lack information that was previously shared\n"
        "5. Include relevant policy details when known\n"
        "6. Review entire conversation history before responding\n"
        "7. Update user identity when new information is provided\n"
        "8. Prioritize user-stated identity over conflicting retrieved information\n"
        "9. For specific policy questions, provide general information and recommend direct contact\n"
        "10. Always mention the importance of consulting with insurance professionals for personalized advice"
    )

    # Build the conversation prompt
    prompt = f"<|system|>\n{system_prompt}\n</s>\n"
    for msg in state["messages"]:
        if isinstance(msg, (HumanMessage, AIMessage)):
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            prompt += f"<|{role}|>\n{msg.content}\n</s>\n"
    if state.get("summary"):
        prompt += f"<|system|>\nPrevious conversation summary: {state['summary']}\nUse this summary to maintain context and consistency in your responses.\n</s>\n"
    prompt += f"<|user|>\n{last_message.content}\n</s>\n<|assistant|>\n"

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
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


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", confid_checker)
graph_builder.add_node("RAG_model", RAG)
graph_builder.add_node("FT_model", FT)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {
        "RAG_model": "RAG_model",
        "FT_model": "FT_model"
    }
)

graph_builder.add_edge("RAG_model", END)
graph_builder.add_edge("FT_model", END)

graph = graph_builder.compile(checkpointer=MemorySaver())

def run_chatbot():
    config = {"configurable": {"thread_id": 1}}
    state = {"messages": [], "message_type": None, "memory": {}, "summary": None}

    print("Chatbot started. Say 'quit' to end the conversation.")
    #    speak("Hello! I'm your insurance assistant. How can I help you today?")
   

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
#                speak(last_message.content)
            else:
                print("No valid response from model.")
                speak("Sorry, I didn't get a response.")

        except Exception as e:
            print(f"Error: {e}")
            speak("Sorry, there was an error. Please try again.")


if __name__ == "__main__":
    run_chatbot()