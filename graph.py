from langgraph.graph import StateGraph, END
from shopping_assistant.state import ProductSearchState

# Node imports
from shopping_assistant.nodes.supervisor import supervisor_node
from shopping_assistant.nodes.guardrails import guardrails_node
from shopping_assistant.nodes.cart_manager import cart_manager_node
from shopping_assistant.nodes.small_talk import small_talk_node
from shopping_assistant.nodes.out_of_domain import out_of_domain_node
from shopping_assistant.nodes.intent_classifier import intent_classifier_node
from shopping_assistant.nodes.sql_agent import sql_agent_node
from shopping_assistant.nodes.entity_extractor import entity_extractor_node
from shopping_assistant.nodes.context_stitcher import conversation_stitcher_node
from shopping_assistant.nodes.vector_search import vector_search_node
from shopping_assistant.nodes.clarification_checker import clarification_checker_node
from shopping_assistant.nodes.response_generator import response_generator_node

# Routing functions
def supervisor_router(state: ProductSearchState) -> str:
    return state["next_action"]

def guardrails_router(state: ProductSearchState) -> str:
    return "continue_processing" if state["is_safe"] else "end_unsafe"

def route_after_intent(state: ProductSearchState) -> str:
    """Route after intent classification - ALL queries go through entity extraction first"""
    # EVERYONE goes through entity extraction for context management
    return "entity_extractor"

# NEW routing function after context stitching
def route_after_context_stitching(state: ProductSearchState) -> str:
    """Route after context has been stitched - determines final destination"""
    intent = state["intent"]
    if intent == "faq":
        return "sql_agent"
    else:
        return "vector_search"

def should_continue(state: ProductSearchState) -> str:
    return "continue" if state["needs_clarification"] else "end"

# Main graph function
def create_product_search_graph():
    graph = StateGraph(ProductSearchState)

    # ADD ALL NODES
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("guardrails", guardrails_node)
    graph.add_node("cart_manager", cart_manager_node)
    graph.add_node("small_talk", small_talk_node)
    graph.add_node("out_of_domain", out_of_domain_node)
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("entity_extractor", entity_extractor_node)
    graph.add_node("conversation_stitcher", conversation_stitcher_node)
    graph.add_node("vector_search", vector_search_node)
    graph.add_node("clarification_checker", clarification_checker_node)
    graph.add_node("response_generator", response_generator_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Routing from supervisor - ALWAYS go through intent_classifier first
    graph.add_conditional_edges("supervisor", supervisor_router, {
        "guardrails": "guardrails",
        "cart_manager": "cart_manager",
        "small_talk": "small_talk", 
        "out_of_domain": "out_of_domain",
        "intent_classifier": "intent_classifier"  # ALL other paths go here
    })

    # Guardrails conditional
    graph.add_conditional_edges("guardrails", guardrails_router, {
        "continue_processing": "intent_classifier",
        "end_unsafe": END
    })

    # Terminal edges for supervisor direct routes
    graph.add_edge("cart_manager", END)
    graph.add_edge("small_talk", END)
    graph.add_edge("out_of_domain", END)

    # Intent classification - ENHANCED to handle context for ALL queries
    graph.add_conditional_edges("intent_classifier", route_after_intent, {
        "sql_agent": "entity_extractor",      # CHANGED: Go through context pipeline first
        "entity_extractor": "entity_extractor"
    })

    # UNIVERSAL context pipeline - ALL queries go through this
    graph.add_edge("entity_extractor", "conversation_stitcher")
    
    # After context stitching, route based on intent
    graph.add_conditional_edges("conversation_stitcher", route_after_context_stitching, {
        "sql_agent": "sql_agent",           # Now go to SQL with context
        "vector_search": "vector_search"    # Or continue to vector search
    })

    # SQL agent now terminates after context processing
    graph.add_edge("sql_agent", END)

    # Product search path continues normally
    graph.add_edge("vector_search", "clarification_checker")
    graph.add_edge("clarification_checker", "response_generator")

    # Final clarification handling
    graph.add_conditional_edges("response_generator", should_continue, {
        "continue": END,
        "end": END
    })

    return graph.compile()

# NEW routing function after context stitching
def route_after_context_stitching(state: ProductSearchState) -> str:
    """Route after context has been stitched - determines final destination"""
    intent = state["intent"]
    if intent == "faq":
        return "sql_agent"
    else:
        return "vector_search"