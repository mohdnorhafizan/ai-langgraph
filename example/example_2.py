from langchain_openai import ChatOpenAI

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

def web_search(query: str) -> str:
    """
    Search for information.

    This is a demo tool. In a real application this would call
    an actual search API.
    """

    print(f"\n[TOOL] web_search('{query}')")

    return """
    2024 FAANG headcount:

    Meta: 74,067
    Apple: 164,000
    Amazon: 1,556,000
    Netflix: 14,000
    Alphabet: 183,323
    """


def company_info(company: str) -> str:
    """
    Return information about a company.
    """

    print(f"\n[TOOL] company_info('{company}')")

    companies = {
        "Meta": "Meta Platforms, Inc.",
        "Apple": "Apple Inc.",
        "Amazon": "Amazon.com, Inc.",
        "Netflix": "Netflix, Inc.",
        "Alphabet": "Alphabet Inc.",
    }

    return companies.get(
        company,
        f"No information found for {company}"
    )


# ============================================================
# 3. MATH TOOLS
# ============================================================

def add(a: float, b: float) -> float:
    """Add two numbers."""

    print(f"\n[TOOL] add({a}, {b})")

    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""

    print(f"\n[TOOL] multiply({a}, {b})")

    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b."""

    print(f"\n[TOOL] divide({a}, {b})")

    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


def percentage(part: float, total: float) -> float:
    """Calculate percentage."""

    print(f"\n[TOOL] percentage({part}, {total})")

    if total == 0:
        raise ValueError("Total cannot be zero.")

    return (part / total) * 100


# ============================================================
# 4. ANALYSIS TOOLS
# ============================================================

def calculate_difference(
    first: float,
    second: float
) -> float:
    """Calculate the absolute difference between two values."""

    print(
        f"\n[TOOL] calculate_difference({first}, {second})"
    )

    return abs(first - second)


def calculate_ratio(
    first: float,
    second: float
) -> float:
    """Calculate first value divided by second value."""

    print(
        f"\n[TOOL] calculate_ratio({first}, {second})"
    )

    if second == 0:
        raise ValueError("Second value cannot be zero.")

    return first / second


# ============================================================
# 5. REVIEW TOOLS
# ============================================================

def validate_number(value: float) -> str:
    """Validate whether a numeric result is reasonable."""

    print(f"\n[TOOL] validate_number({value})")

    if value < 0:
        return "WARNING: result is negative."

    return "VALID: result is a non-negative number."


# ============================================================
# 6. CREATE SPECIALIZED AGENTS
# ============================================================


# ------------------------------------------------------------
# Research Agent
# ------------------------------------------------------------

research_agent = create_react_agent(
    model=model,

    tools=[
        web_search,
        company_info,
    ],

    name="research_expert",

    prompt="""
You are a research expert.

Your responsibilities:
- Find factual information.
- Use web_search when information needs to be retrieved.
- Use company_info when company-specific information is required.
- Do NOT perform mathematical calculations.
- Return structured factual information to the supervisor.
- Clearly identify uncertainty.
"""
)


# ------------------------------------------------------------
# Math Agent
# ------------------------------------------------------------

math_agent = create_react_agent(
    model=model,

    tools=[
        add,
        multiply,
        divide,
        percentage,
    ],

    name="math_expert",

    prompt="""
You are a mathematical expert.

Your responsibilities:
- Perform calculations using tools.
- Do not rely on mental arithmetic when a tool is available.
- Use one mathematical tool at a time.
- Verify important calculations.
- Clearly explain the final numerical result.
"""
)


# ------------------------------------------------------------
# Analysis Agent
# ------------------------------------------------------------

analysis_agent = create_react_agent(
    model=model,

    tools=[
        calculate_difference,
        calculate_ratio,
    ],

    name="analysis_expert",

    prompt="""
You are a data analysis expert.

Your responsibilities:
- Analyze numerical results.
- Compare companies or values.
- Calculate differences and ratios using tools.
- Identify interesting patterns.
- Do not retrieve external information.
- Do not invent missing data.
"""
)


# ------------------------------------------------------------
# Reviewer Agent
# ------------------------------------------------------------

reviewer_agent = create_react_agent(
    model=model,

    tools=[
        validate_number,
    ],

    name="reviewer_expert",

    prompt="""
You are a quality assurance reviewer.

Your responsibilities:
- Review the work produced by other agents.
- Check whether calculations appear internally consistent.
- Check whether research results contain obvious problems.
- Use validate_number when validating numerical results.
- Identify missing information.
- Report whether the result is ready to be presented.
"""
)


# ============================================================
# 7. SUPERVISOR
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
You are the main supervisor coordinating four specialized agents.

AVAILABLE AGENTS:

1. research_expert
   - Retrieves factual information.
   - Searches for company or external information.
   - Does NOT perform mathematical calculations.

2. math_expert
   - Performs mathematical calculations.
   - Uses mathematical tools.
   - Does NOT perform external research.

3. analysis_expert
   - Compares numerical results.
   - Calculates differences and ratios.
   - Performs data analysis.

4. reviewer_expert
   - Reviews the work of other agents.
   - Checks consistency and quality.
   - Identifies missing information or suspicious results.

ROUTING RULES:

- Information retrieval → research_expert
- Mathematical calculation → math_expert
- Comparison or numerical analysis → analysis_expert
- Quality checking → reviewer_expert

IMPORTANT:

A request may require multiple agents.

For example:

1. Research the required information.
2. Send the information to math_expert for calculations.
3. Send the results to analysis_expert for interpretation.
4. Send the final work to reviewer_expert for validation.

Do not assume that one agent must solve the entire request.

You are responsible for deciding the correct sequence.

When the work has been sufficiently completed and reviewed,
return the final answer to the user.
""",

    output_mode="full_history"
)


# ============================================================
# 8. COMPILE
# ============================================================

app = workflow.compile()


# ============================================================
# 9. TEST
# ============================================================

query = """
Find the 2024 headcount of the FAANG companies.

Then:
1. Calculate the combined headcount.
2. Calculate the average headcount.
3. Identify which company has the largest headcount.
4. Calculate how many times larger Amazon's headcount is
   compared with Netflix's.
5. Review the calculations before giving me the final answer.
"""


print("\n")
print("=" * 80)
print("USER REQUEST")
print("=" * 80)
print(query)


result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ]
    }
)


# ============================================================
# 10. FINAL RESULT
# ============================================================

print("\n")
print("=" * 80)
print("FINAL RESULT")
print("=" * 80)

print(
    result["messages"][-1].content
)