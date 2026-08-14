import operator

from typing import Annotated, Literal, Sequence
from langchain_core.messages import AIMessage
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_core.tools import tool

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. Tool
# ============================================================

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


# ============================================================
# 2. State
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[
        Sequence[BaseMessage],
        operator.add
    ]

    next: str


# ============================================================
# 3. Math Expert
# ============================================================

def math_expert_node(state: AgentState):

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    llm_with_tools = llm.bind_tools([add_numbers])

    system_instruction = (
        "You are a math expert. "
        "Always use the add_numbers tool when adding numbers."
    )

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=system_instruction)
        ] + list(state["messages"])
    )

    print("\n=== MATH EXPERT ===")
    print("Content:", response.content)
    print("Tool calls:", response.tool_calls)
    print("Invalid tool calls:", response.invalid_tool_calls)
    print("===================\n")

    return {
        "messages": [response]
    }


def should_continue(state):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "result_node"

def result_node(state: AgentState):
    print("\n=== RESULT NODE ===")

    last_message = state["messages"][-1]

    print("Final math result:")
    print(last_message.content)

    return {
        "messages": [
            AIMessage(
                content=f"Result processed: {last_message.content}",
                name="result_node"
            )
        ]
    }
# ============================================================
# 4. Tool Node
# ============================================================

tool_node = ToolNode([add_numbers])


# ============================================================
# 5. Build Graph
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node(
    "math_expert",
    math_expert_node
)

workflow.add_node(
    "result_node",
    result_node
)

workflow.add_node(
    "tools",
    tool_node
)

workflow.add_edge(
    START,
    "math_expert"
)

# If math expert asks for a tool,
# execute the tool.
workflow.add_conditional_edges(
    "math_expert",
    should_continue,
    {
        "tools": "tools",
        "result_node": "result_node",
    }
)

workflow.add_edge(
    "tools",
    "math_expert"
)

workflow.add_edge(
    "result_node",
    END
)

app = workflow.compile()


# ============================================================
# 6. Run
# ============================================================

initial_state = {
    "messages": [
        HumanMessage(
            content="Add 10 and 20."
        )
    ]
}

for output in app.stream(initial_state):

    print("\n--- STATE DELTA ---")

    for node_name, state_delta in output.items():

        print("Node:", node_name)
        print("Delta:", state_delta)