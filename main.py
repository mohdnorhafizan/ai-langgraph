import operator
import os
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load values from .env into process environment variables
load_dotenv()

# =====================================================================
# 1. State Definition
# =====================================================================

class AgentState(TypedDict):
    """Tracks the total conversation state across all specialized workers."""
    # Messages are appended over time using operator.add
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Tracks the decision made by the supervisor on who goes next
    next: str

# =====================================================================
# 2. Supervisor Schema & Routing Decision Logic
# =====================================================================

class RouterSchema(BaseModel):
    """The structured schema that enforces the supervisor LLM's routing choices."""
    next: Literal["math_expert", "string_expert", "FINISH"] = Field(
        description="The next specialized agent to execute, or 'FINISH' if the user's intent is fully satisfied."
    )

def supervisor_node(state: AgentState):
    """The manager agent that evaluates conversation context and assigns tasks."""
    messages = state["messages"]
    
    # Comprehensive instructions for the supervisor outlining team layout
    system_instruction = (
        "You are an AI Supervisor orchestrating two specialized sub-agents: 'math_expert' and 'string_expert'.\n"
        "Your task is to view the entire dialogue history and delegate the work effectively.\n\n"
        "Rules:\n"
        "1. If the request requires mathematical computation, routing to 'math_expert' is required.\n"
        "2. If the request involves modifying text transformations, text reversing, or formatting, send it to 'string_expert'.\n"
        "3. If a worker has provided an answer and the request is fully handled, return 'FINISH'.\n"
        "Do not answer the user directly. Always delegate tasks or signal completion."
    )
    
    # Connect to OpenAI and force structural validation matching the RouterSchema
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(RouterSchema)
    
    # Construct complete payload for evaluation
    full_history = [SystemMessage(content=system_instruction)] + list(messages)
    routing_decision = structured_llm.invoke(full_history)
    
    # Pass selection forward through the graph's shared state
    return {"next": routing_decision.next}

# =====================================================================
# 3. Specialized Worker Agent Nodes
# =====================================================================

def math_expert_node(state: AgentState):
    """A worker node specialized entirely in logical or mathematical computations."""
    messages = state["messages"]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    worker_instruction = (
        "You are a math expert. Compute the solution requested by the user. "
        "Show step-by-step arithmetic details clearly and concisely."
    )
    
    response = llm.invoke([SystemMessage(content=worker_instruction)] + list(messages))
    
    # Append the worker response to the message stream under its unique name identifier
    return {
        "messages": [AIMessage(content=response.content, name="math_expert")]
    }

def string_expert_node(state: AgentState):
    """A worker node specialized in handling string modifications and linguistics."""
    messages = state["messages"]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    worker_instruction = (
        "You are a text processing expert. Handle word adjustments, modifications, character counting, "
        "or reversals requested by the user. Be precise and elegant."
    )
    
    response = llm.invoke([SystemMessage(content=worker_instruction)] + list(messages))
    
    return {
        "messages": [AIMessage(content=response.content, name="string_expert")]
    }

# =====================================================================
# 4. Constructing and Orchestrating the StateGraph
# =====================================================================

# Initialize the workflow graph with our tracking state layout
workflow = StateGraph(AgentState)

# Register the architectural structural elements as graph nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("math_expert", math_expert_node)
workflow.add_node("string_expert", string_expert_node)

# Workers must always hand execution control directly back to the manager node
workflow.add_edge("math_expert", "supervisor")
workflow.add_edge("string_expert", "supervisor")

# The supervisor dictates workflow path routing dynamically using conditional edges
routing_map = {
    "math_expert": "math_expert",
    "string_expert": "string_expert",
    "FINISH": END
}

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],  # Evaluates the string set inside state["next"]
    routing_map
)

# Every transaction loop starts directly at the supervisor node
workflow.set_entry_point("supervisor")

# Compile everything into an executable application graph
app = workflow.compile()

# =====================================================================
# 5. Local Script Execution Example
# =====================================================================

if __name__ == "__main__":
    # Quick configuration validation check before trying to invoke OpenAI dependencies
    if not os.getenv("OPENAI_API_KEY"):
        print("Please ensure your OPENAI_API_KEY environment variable is exported before execution.")
    else:
        # Scenario requiring multi-agent handoffs sequentially:
        # 1. Math expert processes numerical calculation
        # 2. String expert takes calculation and formats text layout
        sample_prompt = "Calculate 143 multiplied by 7, and then reverse that final number sequence out as a string text."
        
        print(f"Submitting query: '{sample_prompt}'\n")
        
        initial_state = {
            "messages": [HumanMessage(content=sample_prompt)]
        }
        
        # Stream the full execution cycle sequence events cleanly into terminal output
        for output in app.stream(initial_state):
            for node_name, state_delta in output.items():
                print(f"--- Node Executed: {node_name} ---")
                if "next" in state_delta:
                    print(f"Supervisor choice for next target node: '{state_delta['next']}'")
                if "messages" in state_delta:
                    last_msg = state_delta["messages"][-1]
                    print(f"Output contents:\n{last_msg.content}")
                print("-" * 40)