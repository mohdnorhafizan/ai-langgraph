from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 1. MODEL
# ============================================================

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)


# ============================================================
# 2. RESEARCH TOOLS
# ============================================================

def get_company_headcount(company: str, year: int) -> str:
    """
    Get the employee headcount for a company in a specific year.

    Use this when factual employee/headcount information
    is required.
    """

    data = {
        ("Meta", 2024): 74067,
        ("Apple", 2024): 164000,
        ("Amazon", 2024): 1556000,
        ("Netflix", 2024): 14000,
        ("Alphabet", 2024): 183323,
    }

    value = data.get((company, year))

    if value is None:
        return f"No headcount data available for {company} in {year}."

    return f"{company} had {value} employees in {year}."


def get_company_list() -> str:
    """
    Return the companies available in the dataset.
    """

    return """
    Available companies:

    - Meta
    - Apple
    - Amazon
    - Netflix
    - Alphabet
    """


# ============================================================
# 3. MATH TOOLS
# ============================================================

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


# ============================================================
# 4. ANALYSIS TOOLS
# ============================================================

def calculate_percentage_change(
    old_value: float,
    new_value: float
) -> float:
    """
    Calculate percentage change between two values.
    """

    if old_value == 0:
        raise ValueError("Old value cannot be zero.")

    return ((new_value - old_value) / old_value) * 100


def calculate_ratio(
    value_a: float,
    value_b: float
) -> float:
    """
    Calculate the ratio between two values.
    """

    if value_b == 0:
        raise ValueError("Second value cannot be zero.")

    return value_a / value_b


# ============================================================
# 5. REVIEW TOOL
# ============================================================

def validate_result(
    result: float
) -> str:
    """
    Perform a basic validation of a numerical result.
    """

    if result < 0:
        return "WARNING: numerical result is negative."

    return "PASS: numerical result is valid."


# ============================================================
# 6. PLANNER AGENT
# ============================================================

planner_agent = create_react_agent(
    model=model,

    tools=[],

    name="planner",

    prompt="""
You are a planning expert.

Your job is NOT to solve the user's task.

Your job is to understand the user's goal and create a
high-level plan that identifies:

1. What information is required.
2. What type of work is required.
3. Which specialist capabilities may be needed.
4. What the expected final result should contain.

Do not invent information.

Do not perform calculations.

Keep the plan concise.

The plan should describe WHAT needs to be accomplished,
not provide a rigid implementation sequence.
"""
)


# ============================================================
# 7. RESEARCH AGENT
# ============================================================

research_agent = create_react_agent(
    model=model,

    tools=[
        get_company_headcount,
        get_company_list,
    ],

    name="research_expert",

    prompt="""
You are a research specialist.

Your responsibility is to retrieve factual information
using the available tools.

Rules:

- Use tools when factual information is required.
- Do not invent data.
- Do not perform complex mathematical calculations.
- Clearly identify missing information.
- Return factual findings that other agents can use.

Your output should make it easy for another agent to
perform calculations or analysis.
"""
)


# ============================================================
# 8. MATH AGENT
# ============================================================

math_agent = create_react_agent(
    model=model,

    tools=[
        add,
        divide,
        multiply,
    ],

    name="math_expert",

    prompt="""
You are a mathematics specialist.

Your responsibility is to perform numerical calculations.

Rules:

- Use the available mathematical tools.
- Do not invent input values.
- Do not perform research.
- Verify important calculations.
- Clearly provide the numerical result.
"""
)


# ============================================================
# 9. ANALYSIS AGENT
# ============================================================

analysis_agent = create_react_agent(
    model=model,

    tools=[
        calculate_percentage_change,
        calculate_ratio,
    ],

    name="analysis_expert",

    prompt="""
You are a data analysis specialist.

Your responsibility is to interpret numerical results.

You can:

- compare values
- calculate ratios
- calculate percentage changes
- identify significant differences
- explain patterns

Rules:

- Use tools when calculations are required.
- Do not invent data.
- Do not perform external research.
- Clearly distinguish facts from interpretation.
"""
)


# ============================================================
# 10. REVIEWER AGENT
# ============================================================

reviewer_agent = create_react_agent(
    model=model,

    tools=[
        validate_result,
    ],

    name="reviewer",

    prompt="""
You are a quality assurance reviewer.

Your responsibility is to review work produced by other
agents.

Check:

- Are the required questions answered?
- Are calculations internally consistent?
- Is any information missing?
- Are there unsupported assumptions?
- Are numerical results reasonable?

Use the validation tool when appropriate.

If something is wrong or incomplete, clearly explain
what needs to be corrected.

If everything is satisfactory, say that the work is ready.
"""
)


# ============================================================
# 11. SUPERVISOR
# ============================================================

workflow = create_supervisor(
    [
        research_agent,
        math_agent,
        analysis_agent,
        reviewer_agent,
    ],

    model=model,

    prompt="""
You are the main supervisor.

Your job is to coordinate specialist agents to satisfy
the user's goal.

You have access to:

- research_expert
- math_expert
- analysis_expert
- reviewer

The user may provide a plan from another planning step,
but the plan is guidance rather than a rigid workflow.

Your responsibilities:

1. Understand the user's actual goal.
2. Determine what information or work is still required.
3. Delegate work to the appropriate specialist.
4. Decide when additional specialist work is necessary.
5. Send work to the reviewer when validation is useful.
6. If the reviewer identifies a problem, delegate corrective
   work to the appropriate specialist.
7. Do not perform specialist calculations yourself when the
   math agent is available.
8. Do not invent information.
9. Finish only when the user's goal has been adequately
   satisfied.

IMPORTANT:

You do NOT need to use every agent.

Use only the agents necessary for the request.

For example:

A simple calculation may only require math_expert.

A factual question may only require research_expert.

A complex data question may require research_expert,
math_expert, analysis_expert, and reviewer.

Choose the appropriate workflow dynamically.
"""
)


# ============================================================
# 12. COMPILE
# ============================================================

app = workflow.compile()


# ============================================================
# 13. USER REQUEST
# ============================================================

user_request = """
Analyze the 2024 FAANG employee headcount.

I want to know:

- the total headcount
- the average headcount
- which company has the largest workforce
- how many times larger Amazon is than Netflix
- whether the final calculations are consistent

Explain the result clearly.
"""


# ============================================================
# 14. STEP 1 — CREATE A PLAN
# ============================================================

print("\n")
print("=" * 80)
print("STEP 1 - PLANNER")
print("=" * 80)

plan_result = planner_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_request,
            }
        ]
    }
)

plan = plan_result["messages"][-1].content

print(plan)


# ============================================================
# 15. STEP 2 — SEND GOAL + PLAN TO SUPERVISOR
# ============================================================

supervisor_request = f"""
USER REQUEST:

{user_request}


PLANNER'S ANALYSIS:

{plan}


Use the planner's analysis as guidance.

Determine the appropriate specialist workflow and
complete the user's request.
"""


print("\n")
print("=" * 80)
print("STEP 2 - SUPERVISOR")
print("=" * 80)

result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": supervisor_request,
            }
        ]
    }
)


# ============================================================
# 16. FINAL RESULT
# ============================================================

import json

final_message = result["messages"][-1]

final_result = {
    "content": final_message.content
}

print("\n")
print("=" * 80)
print("FINAL RESULT")
print("=" * 80)

print(
    json.dumps(final_result, indent=4)
)