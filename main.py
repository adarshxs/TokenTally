from sidebar import sidebar
from overview import display_overview
from llm_cost_calculator import display_llm_cost_tool
from transformer_memory_calculator import display_transformer_memory_tool
from llm_recomender import display_llm_recomender_tool

def main():
    
    selected_product = sidebar()
    
    if selected_product == "Overview":
        display_overview()
    elif selected_product == "LLM Cost Tool":
        display_llm_cost_tool()
    elif selected_product == "Transformer Memory Tool":
        display_transformer_memory_tool()
    elif selected_product == "LLM Model Recommendation":
        display_llm_recomender_tool()

if __name__ == "__main__":
    main()
