from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, trim_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, StateGraph, START, END

load_dotenv()

memory = MemorySaver()
model = ChatOpenAI(model_name='gpt-4o')

@tool
def plus(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@tool
def minus(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    return a / b

@tool
def root(a: int) -> float:
    """Get the square root of a number"""
    return a ** 0.5

@tool
def power(a: int, b: int) -> int:
    """Raise a number to a power"""
    return a ** b


tools = [plus, minus, multiply, divide, root, power]

tool_node = ToolNode(tools)

bound_model = model.bind_tools(tools)


def should_continue(state: MessagesState):
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return END

    return "action"

def call_model(state:MessagesState):
    messages = filter_messages().invoke(state["messages"])
    response = bound_model.invoke(messages)

    return {"messages": response}

def filter_messages():
    return trim_messages(
        max_tokens=10,
        token_counter=len,
        strategy="last",
        include_system=True,
        allow_partial=False
    )

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")

workflow.add_conditional_edges("agent", should_continue, ["action", END])
workflow.add_edge("action", "agent")

app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="What is 2 + 2 /2 * 3 -1?")

for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()


