"""
Simple LangGraph Agent Example using Ollama with a Calculator Tool

This example demonstrates:
- Setting up a LangGraph agent with Ollama
- Creating a simple calculator tool
- Defining the agent workflow graph
- Running the agent with tool calling capabilities

Requirements:
pip install langgraph langchain-ollama langchain-core
"""

import json
import operator
from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# Define the state structure for our agent
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# Create a simple calculator tool
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        The result of the calculation as a string
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        # Evaluate the expression
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# Create a simple addition tool as another example
@tool
def add_numbers(a: float, b: float) -> float:
    """
    Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b


# Initialize Ollama model (make sure you have a model pulled, e.g., llama3.1)
llm = ChatOllama(
    model="gpt-oss:20b",  # Change this to your preferred model
    temperature=0,
    base_url="http://192.168.50.159:11434"  # Default Ollama URL
)

# Bind tools to the model
tools = [calculator, add_numbers]
llm_with_tools = llm.bind_tools(tools)


def should_continue(state: AgentState) -> Literal["tools", END]:
    """Determine whether to continue to tools or end the conversation."""
    messages = state['messages']
    last_message = messages[-1]

    # If the LLM makes a tool call, then we route to the "tools" node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


def call_model(state: AgentState) -> dict:
    """Call the language model with the current state."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Create the tool node
tool_node = ToolNode(tools)

# Build the agent graph
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# Add edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()


def run_agent(user_input: str) -> str:
    """Run the agent with a user input and return the final response."""
    initial_state = {"messages": [HumanMessage(content=user_input)]}

    print(f"User: {user_input}")
    print("-" * 50)

    # Run the agent
    for output in app.stream(initial_state):
        for key, value in output.items():
            if key == "agent":
                message = value["messages"][-1]
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    print(f"Agent is calling tools: {[tc['name'] for tc in message.tool_calls]}")
                else:
                    print(f"Agent: {message.content}")
            elif key == "tools":
                tool_messages = value["messages"]
                for msg in tool_messages:
                    if isinstance(msg, ToolMessage):
                        print(f"Tool result: {msg.content}")
        print("-" * 30)

    # Get the final state to return the last message
    final_state = app.invoke(initial_state)
    return final_state["messages"][-1].content


# Example usage


# Interactive mode function
def interactive_mode():
    """Run the agent in interactive mode."""
    print("=== Interactive Mode ===")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        try:
            response = run_agent(user_input)
            print(f"\nFinal response: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")

# Uncomment the line below to run in interactive mode
# interactive_mode()

if __name__ == "__main__":
    # Test the agent with different types of queries

    print("=== LangGraph Agent with Ollama Demo ===\n")

    interactive_mode()
    #
    # # Example 1: Simple calculation
    # response1 = run_agent("What is 15 * 8 + 32?")
    # print(f"Final response: {response1}\n")
    #
    # # Example 2: Using the add_numbers tool
    # response2 = run_agent("Can you add 25.5 and 17.3 for me?")
    # print(f"Final response: {response2}\n")
    #
    # # Example 3: Complex calculation
    # response3 = run_agent("Calculate the result of (100 - 25) * 3 / 5")
    # print(f"Final response: {response3}\n")
    #
    # # Example 4: Regular conversation (no tools needed)
    # response4 = run_agent("Hello! How are you today?")
    # print(f"Final response: {response4}\n")
