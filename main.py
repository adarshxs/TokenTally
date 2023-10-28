from sidebar import sidebar
from llm_cost_calculator import display_llm_cost_tool
from transformer_memory_calculator import display_transformer_memory_tool

def main():
    
    sidebar()
    
    display_llm_cost_tool()
    display_transformer_memory_tool()

if __name__ == "__main__":
    main()