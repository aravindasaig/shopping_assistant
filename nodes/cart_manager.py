from shopping_assistant.state import ProductSearchState
from shopping_assistant.schema import CartItem, ShoppingCart
from langchain_core.messages import AIMessage
import json

def cart_manager_node(state: ProductSearchState) -> ProductSearchState:
    print("🛒 Handling cart operation...")

    cart = state.get("shopping_cart", ShoppingCart())
    user_input = state["user_input"].lower()
    results = state.get("search_results", [])
    memory = state["conversation_memory"]

    # Check for ADD actions (most flexible)
    if any(word in user_input for word in ["add", "buy", "get", "want", "purchase", "take"]):
        # Additional context clues for adding
        if any(phrase in user_input for phrase in ["to cart", "this", "it", "1st", "first", "second", "third", "last"]):
            action = "add"
        elif any(word in user_input for word in ["add", "buy", "get", "purchase"]):
            action = "add"
    
    # Check for REMOVE actions
    elif any(word in user_input for word in ["remove", "delete"]):
        action = "remove"
    
    # Check for CHECKOUT actions
    elif any(word in user_input for word in ["checkout", "payment", "proceed"]):
        action = "checkout"
    
    # Check for VIEW actions
    elif any(phrase in user_input for phrase in ["show cart", "view cart", "my cart", "cart summary", "what's in my cart"]):
        action = "view"
    
    # Default fallback
    else:
        action = "view"

    print(f"🔍 Detected cart action: {action}")
    response = ""

    if action == "add":
        # Fallback to last successful search if no recent results
        if not results:
            for turn in reversed(memory.turn_history):
                if turn.search_results:
                    results = turn.search_results
                    print(f"📚 Using search results from turn {turn.turn_id}")
                    break

        if results:
            # Determine which item to add with better parsing
            item_index = 0
            
            # Parse item selection more flexibly
            if any(word in user_input for word in ["1st", "first", "1"]):
                item_index = 0
            elif any(word in user_input for word in ["2nd", "second", "2"]):
                item_index = 1
            elif any(word in user_input for word in ["3rd", "third", "3"]):
                item_index = 2
            elif any(word in user_input for word in ["4th", "fourth", "4"]):
                item_index = 3
            elif any(word in user_input for word in ["5th", "fifth", "5"]):
                item_index = 4
            elif "last" in user_input:
                item_index = len(results) - 1

            item = results[min(item_index, len(results) - 1)]
            
            # ENHANCED: Use standardized metadata structure
            if "metadata" in item and isinstance(item["metadata"], dict):
                # New standardized structure
                metadata = item["metadata"]
                print(f"✅ Using standardized metadata structure")
            else:
                # Legacy structure - extract manually
                print(f"🔄 Converting legacy structure to standardized metadata")
                metadata = extract_legacy_metadata(item)
            
            print(f"🔍 Final metadata: {metadata}")

            try:
                cart_item = CartItem(
                    product_id=metadata.get("image_id", f"item_{item_index}"),
                    product_name=metadata.get("product_type", "Product"),
                    brand=metadata.get("brand", "Unknown"),
                    color=metadata.get("color", "N/A"),
                    price=float(metadata.get("price_inr", 0.0))
                )
                cart.add_item(cart_item)
                response = f"✅ Added {cart_item.brand} {cart_item.product_name} ({cart_item.color}) - ₹{cart_item.price} to your cart!\n\n{cart.get_summary()}"
                print(f"✅ Successfully created cart item: {cart_item.brand} {cart_item.product_name}")
            except Exception as e:
                print(f"❌ Failed to create cart item: {e}")
                response = "❌ Couldn't add item to cart. Please try again."

        else:
            response = "I couldn't find any products to add. Please search for items first."

    elif action == "view":
        response = cart.get_summary()

    elif action == "remove":
        if cart.items:
            removed = cart.items[-1]
            cart.remove_item(removed.product_name)
            response = f"🗑️ Removed {removed.product_name} from your cart.\n\n{cart.get_summary()}"
        else:
            response = "🛒 Your cart is already empty."

    elif action == "checkout":
        if cart.items:
            response = (
                f"🎉 You're ready to checkout!\n\n{cart.get_summary()}\n\n"
                "To complete your order:\n"
                "1. Confirm items\n"
                "2. Enter shipping info\n"
                "3. Proceed to payment"
            )
        else:
            response = "🛒 Your cart is empty. Add something before checking out!"

    state["shopping_cart"] = cart
    state["cart_action"] = action
    state["agent_response"] = response
    state["messages"].append(AIMessage(content=response))

    return state

def extract_legacy_metadata(item: dict) -> dict:
    """Convert legacy search result structure to standardized metadata"""
    metadata = {}
    
    # Try multiple extraction methods
    if 'properties' in item:
        props = item['properties']
        raw_metadata = props.get('metadata', {})
    elif 'data' in item:
        raw_metadata = item['data'].get('metadata', {})
    elif 'metadata' in item:
        raw_metadata = item['metadata']
    else:
        raw_metadata = {}
    
    # Parse JSON string if needed
    if isinstance(raw_metadata, str):
        try:
            metadata = json.loads(raw_metadata)
        except:
            metadata = {}
    else:
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    
    return metadata