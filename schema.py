from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict


@dataclass
class CartItem:
    """Individual cart item"""
    product_id: str
    product_name: str
    brand: str
    color: str
    price: float
    quantity: int = 1


@dataclass
class ShoppingCart:
    """Session shopping cart"""
    items: List[CartItem] = field(default_factory=list)
    total_amount: float = 0.0

    def add_item(self, item: CartItem):
        for existing in self.items:
            if (
                existing.product_name == item.product_name and
                existing.brand == item.brand and
                existing.color == item.color
            ):
                existing.quantity += item.quantity
                self._calculate_total()
                return
        self.items.append(item)
        self._calculate_total()

    def remove_item(self, product_name: str):
        self.items = [item for item in self.items if item.product_name != product_name]
        self._calculate_total()

    def _calculate_total(self):
        self.total_amount = sum(item.price * item.quantity for item in self.items)

    def get_summary(self) -> str:
        if not self.items:
            return "🛒 Your cart is empty"

        summary = f"🛒 Cart Summary ({len(self.items)} items):\n\n"
        for i, item in enumerate(self.items, 1):
            summary += (
                f"{i}. {item.brand} {item.product_name} - {item.color}\n"
                f"   ₹{item.price} x {item.quantity} = ₹{item.price * item.quantity}\n\n"
            )
        summary += f"💰 Total: ₹{self.total_amount:,.2f}"
        return summary


@dataclass
class ConversationTurn:
    """Single turn of conversation"""
    turn_id: int
    user_input: str
    extracted_entities: Dict
    timestamp: datetime
    search_results: List[Dict] = field(default_factory=list)
    context_snapshot: Dict = field(default_factory=dict)


@dataclass
class ConversationMemory:
    """Memory of the conversation session"""
    active_context: Dict = field(default_factory=dict)
    turn_history: List[ConversationTurn] = field(default_factory=list)
    successful_searches: List[Dict] = field(default_factory=list)
    clarification_count: int = 0

    def add_turn(self, turn: ConversationTurn):
        self.turn_history.append(turn)

    def get_last_entities(self) -> Dict:
        return self.turn_history[-1].extracted_entities if self.turn_history else {}

    def get_last_successful_search(self) -> Dict:
        for turn in reversed(self.turn_history):
            if turn.search_results:
                return turn.extracted_entities
        return {}
