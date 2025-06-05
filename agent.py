from shopping_assistant.graph import create_product_search_graph
from shopping_assistant.schema import ShoppingCart, ConversationMemory
from shopping_assistant.state import ProductSearchState
from langchain_core.messages import HumanMessage
from typing import Optional


class ProductSearchAgent:
    def __init__(self):
        self.app = create_product_search_graph()
        self.shopping_cart = ShoppingCart()
        self.conversation_memory = ConversationMemory()
        self.turn_count = 0

    def chat(self, user_input: str, image_path: Optional[str] = None) -> str:
        self.turn_count += 1
        print(f"\n🔄 Turn {self.turn_count} | User: {user_input}")

        initial_state: ProductSearchState = {
            "user_input": user_input,
            "has_image": bool(image_path),
            "image_path": image_path,

            # Intent & processing
            "intent": "",
            "intent_confidence": 0.0,
            "raw_entities": {},
            "stitched_entities": {},
            "search_results": [],

            # SQL
            "sql_query": "",
            "sql_results": "",

            # Cart
            "shopping_cart": self.shopping_cart,
            "cart_action": "",
            "selected_product": {},

            # Safety
            "is_safe": True,
            "safety_issues": [],
            "is_in_domain": True,
            "domain_confidence": 1.0,

            # Memory & flow
            "conversation_memory": self.conversation_memory,
            "turn_count": self.turn_count,
            "needs_clarification": False,
            "clarification_question": "",

            # Supervisor
            "next_action": "",
            "supervisor_reasoning": "",

            # Final response
            "agent_response": "",
            "messages": [HumanMessage(content=user_input)]
        }

        final_state = self.app.invoke(initial_state)

        # Sync memory and cart back
        self.shopping_cart = final_state["shopping_cart"]
        self.conversation_memory = final_state["conversation_memory"]

        return final_state["agent_response"]

    def reset_conversation(self):
        self.shopping_cart = ShoppingCart()
        self.conversation_memory = ConversationMemory()
        self.turn_count = 0
        print("🔄 Conversation and cart reset.")
