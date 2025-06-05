from typing import TypedDict, List, Dict, Optional, Annotated
from langchain_core.messages import BaseMessage
from shopping_assistant.schema import ShoppingCart, ConversationMemory
import operator


class ProductSearchState(TypedDict):
    # User input
    user_input: str
    has_image: bool
    image_path: Optional[str]

    # Processing state
    intent: str
    intent_confidence: float
    raw_entities: Dict
    stitched_entities: Dict
    search_results: List[Dict]

    # SQL support
    sql_query: str
    sql_results: str

    # Cart management
    shopping_cart: ShoppingCart
    cart_action: str
    selected_product: Dict

    # Safety & Quality
    is_safe: bool
    safety_issues: List[str]
    is_in_domain: bool
    domain_confidence: float

    # Conversation management
    conversation_memory: ConversationMemory
    turn_count: int
    needs_clarification: bool
    clarification_question: str

    # Supervisor decisions
    next_action: str
    supervisor_reasoning: str

    # Final response
    agent_response: str
    messages: Annotated[List[BaseMessage], operator.add]
