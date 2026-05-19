from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

#----- Tools -----

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f">> Executing get_product_price(product='{product}')")
    prices = {"laptop":1299.99, "headphones":149.95, "keyboard":89.50}
    return prices.get(product, 0)



@tool
def apply_discount(price:float, discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)


# ----- Agent Loop -----

@traceable(name="Langchain Agent Loop")
#ReAct Loop
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name : t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("-" * 60)

    messages = [
        SystemMessage(content="You are a helpful shopping assistant" 
                      "You have access to a product catalog tool and a discount tool" 
                      "\nSTRICT RULES - you MUST follow these exactly\n" \
                      "1. NEVER guess or assume any product price." "" \
                      "\n2. You must call the get_product_price tool first to get the real price." 
                      "\n3. Only call apply_discount AFTER you have received a price from get_product_price. "
                      "\nPass the exact price returned bu get_product_price - do NOT pass made up numbers." 
                      "\n4. NEVER calculate discounts using math. Always use the apply_discount tool."
                      "\n5. If the user does not specify a discount tier, ask them which tier to use - do not assume one."),
        HumanMessage(content=question),

    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n---Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content


if __name__ == "__main__":
    print("Hello Langchain Agent!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")