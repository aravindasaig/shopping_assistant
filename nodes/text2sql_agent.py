from shopping_assistant.state import ProductSearchState
from shopping_assistant.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION
from langchain_core.messages import AIMessage
import os
import sqlite3
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from openai import AzureOpenAI

# Use the same client configuration as other nodes
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
)

@dataclass
class SQLResult:
    """Result of SQL query execution"""
    success: bool
    data: List[Dict[str, Any]]
    error: Optional[str] = None
    sql_query: str = ""
    execution_time: float = 0.0

class Text2SQLAgent:
    """Text to SQL agent for retail database queries"""
    
    def __init__(self, db_path: str = "retail.db"):
        self.db_path = db_path
        self.schema_info = self._build_schema_info()
        
    def _build_schema_info(self) -> str:
        """Build comprehensive schema information for GPT-4.1"""
        schema_info = """
        # RETAIL DATABASE SCHEMA
        
        ## TABLES OVERVIEW:
        
        ### 1. CATEGORIES
        - categories (category_id, category_name)
        - Main product categories like "Fashion", "Electronics", etc.
        
        ### 2. SUBCATEGORIES  
        - subcategories (subcategory_id, category_id, subcategory_name)
        - Linked to categories: "Men's Clothing", "Women's Clothing", "TVs", etc.
        
        ### 3. PRODUCT_TYPES
        - product_types (product_type_id, subcategory_id, product_type_name)
        - Specific product types: "T-shirt", "Jeans", "LED TV", etc.
        
        ### 4. PRODUCTS (Main table)
        - products (product_id, product_type_id, product_name, brand, gender, price_inr, image_path)
        - Contains all products with basic info
        
        ### 5. TSHIRT_ATTRIBUTES (For t-shirts only)
        - tshirt_attributes (product_id, color, pattern, sleeve_type, neck_type, fit, material, theme, visual_tags, occlusion)
        - Additional attributes specific to t-shirts
        
        ### 6. TV_ATTRIBUTES (For TVs only)
        - tv_attributes (product_id, screen_size, resolution, display_type, smart_tv, os, ports, design, stand_type, visual_tags, occlusion)
        - Additional attributes specific to TVs
        
        ## KEY RELATIONSHIPS:
        - categories -> subcategories -> product_types -> products
        - products -> tshirt_attributes (for t-shirts)
        - products -> tv_attributes (for TVs)
        
        ## COMMON QUERIES:
        - Product searches by category, brand, price range
        - T-shirt specific: color, material, fit, sleeve type
        - TV specific: screen size, resolution, smart features
        - Aggregations: count by brand, average price, etc.
        """
        return schema_info
    
    def _validate_sql_safety(self, sql_query: str) -> bool:
        """Validate SQL query for safety"""
        sql_lower = sql_query.lower().strip()
        
        # Only allow SELECT statements
        if not sql_lower.startswith('select'):
            return False
            
        # Disallow dangerous operations
        dangerous_keywords = [
            'drop', 'delete', 'update', 'insert', 'alter', 
            'create', 'truncate', 'replace', 'exec', 'execute'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                return False
                
        return True
    
    def _execute_sql(self, sql_query: str) -> SQLResult:
        """Execute SQL query safely"""
        if not self._validate_sql_safety(sql_query):
            return SQLResult(
                success=False,
                data=[],
                error="Query not allowed - only SELECT statements permitted",
                sql_query=sql_query
            )
        
        try:
            import time
            start_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Return rows as dictionaries
                cursor = conn.cursor()
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                data = [dict(row) for row in rows]
                
                execution_time = time.time() - start_time
                
                return SQLResult(
                    success=True,
                    data=data,
                    sql_query=sql_query,
                    execution_time=execution_time
                )
                
        except Exception as e:
            return SQLResult(
                success=False,
                data=[],
                error=str(e),
                sql_query=sql_query
            )
    
    def natural_language_to_sql(self, question: str) -> str:
        """Convert natural language question to SQL"""
        
        system_prompt = f"""
        You are an expert SQL query generator for a retail database.
        
        SCHEMA INFORMATION:
        {self.schema_info}
        
        RULES:
        1. Generate ONLY SELECT statements
        2. Use proper JOINs to connect related tables
        3. Use appropriate WHERE clauses for filtering
        4. Include relevant columns in SELECT
        5. Use LIMIT when appropriate to avoid huge results
        6. Handle price queries with proper numeric comparisons
        7. Use LIKE for partial text matching
        8. Return valid SQLite syntax
        
        EXAMPLES:
        
        Q: "Show me all t-shirts"
        A: SELECT p.product_name, p.brand, p.price_inr, t.color, t.material 
           FROM products p 
           JOIN product_types pt ON p.product_type_id = pt.product_type_id
           JOIN tshirt_attributes t ON p.product_id = t.product_id
           WHERE pt.product_type_name = 'T-shirt'
           
        Q: "What brands sell t-shirts?"
        A: SELECT DISTINCT p.brand 
           FROM products p 
           JOIN product_types pt ON p.product_type_id = pt.product_type_id
           WHERE pt.product_type_name = 'T-shirt' AND p.brand IS NOT NULL
           
        Q: "How many products under 1000 rupees?"
        A: SELECT COUNT(*) as product_count 
           FROM products 
           WHERE price_inr < 1000
           
        Q: "Average price of Nike products"
        A: SELECT AVG(price_inr) as avg_price 
           FROM products 
           WHERE brand = 'Nike'
           
        Return ONLY the SQL query, no explanations.
        """
        
        user_prompt = f"Generate SQL for: {question}"
        
        try:
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            elif sql_query.startswith("```"):
                sql_query = sql_query.replace("```", "").strip()
                
            return sql_query
            
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return ""
    
    def format_results(self, result: SQLResult, original_question: str) -> str:
        """Format SQL results into natural language"""
        
        if not result.success:
            return f"❌ Query failed: {result.error}"
            
        if not result.data:
            return "📊 No results found for your query."
            
        data_count = len(result.data)
        
        # Format based on result type
        if data_count == 1 and len(result.data[0]) == 1:
            # Single value result (like COUNT, AVG)
            key, value = list(result.data[0].items())[0]
            
            # Handle None values safely
            if value is None:
                return f"📊 {original_question}: No data available"
            elif isinstance(value, float):
                return f"📊 {original_question}: ₹{value:,.2f}"
            else:
                return f"📊 {original_question}: {value:,}"
                
        elif data_count <= 10:
            # Small result set - show all
            formatted_response = f"📊 Found {data_count} results:\n\n"
            
            for i, row in enumerate(result.data, 1):
                formatted_response += f"{i}. "
                
                # Format each row nicely with None handling
                row_parts = []
                for key, value in row.items():
                    if value is not None:
                        if key == 'price_inr':
                            row_parts.append(f"₹{value}")
                        elif key in ['product_name', 'brand', 'color', 'material']:
                            row_parts.append(str(value))
                        elif 'count' in key.lower() or 'avg' in key.lower():
                            if isinstance(value, float):
                                row_parts.append(f"{value:,.2f}")
                            else:
                                row_parts.append(f"{value:,}")
                        else:
                            row_parts.append(str(value))
                    else:
                        # Handle None values
                        if key == 'price_inr':
                            row_parts.append("₹N/A")
                        else:
                            row_parts.append("N/A")
                
                formatted_response += " | ".join(row_parts) + "\n"
                
            return formatted_response
            
        else:
            # Large result set - summarize
            sample_rows = result.data[:5]
            formatted_response = f"📊 Found {data_count} results (showing first 5):\n\n"
            
            for i, row in enumerate(sample_rows, 1):
                formatted_response += f"{i}. "
                row_parts = []
                for key, value in row.items():
                    if value is not None:
                        if key == 'price_inr':
                            row_parts.append(f"₹{value}")
                        elif key in ['product_name', 'brand', 'color', 'material']:
                            row_parts.append(str(value))
                        else:
                            row_parts.append(str(value))
                    else:
                        # Handle None values
                        if key == 'price_inr':
                            row_parts.append("₹N/A")
                        else:
                            row_parts.append("N/A")
                            
                row_parts = row_parts[:3]  # Limit to 3 fields for readability
                formatted_response += " | ".join(row_parts) + "\n"
                
            formatted_response += f"\n... and {data_count - 5} more results."
            return formatted_response
    
    def query(self, question: str, debug: bool = False) -> dict:
        """Main method to answer natural language questions"""
    
        print(f"🤔 Question: {question}")
        
        # Generate SQL
        sql_query = self.natural_language_to_sql(question)
        
        if not sql_query:
            return {
                "query": "",
                "answer": "❌ Could not generate SQL query for your question."
            }
            
        if debug:
            print(f"🔍 Generated SQL: {sql_query}")
        
        # Execute SQL
        result = self._execute_sql(sql_query)
        
        if debug:
            print(f"⏱️ Execution time: {result.execution_time:.3f}s")
            print(f"📊 Rows returned: {len(result.data)}")
        
        # Format results
        formatted_response = self.format_results(result, question)
        
        return {
            "query": sql_query,
            "answer": formatted_response
        }

# SQL Agent Node for LangGraph integration
def sql_agent_node(state: ProductSearchState) -> ProductSearchState:
    """SQL Agent Node: Handle FAQ/analytics queries with context awareness"""
    print("🗃️ Processing FAQ query with Text2SQL...")
    
    # ENHANCEMENT: Context-aware query processing
    user_input = state["user_input"]
    memory = state["conversation_memory"]
    
    # Resolve contextual references using conversation history
    resolved_query = resolve_contextual_references(user_input, memory)
    
    try:
        # Initialize SQL agent
        sql_agent = Text2SQLAgent()
        
        # Process the resolved query with debug enabled
        result = sql_agent.query(resolved_query, debug=True)
        
        # Ensure result is always a dict
        if not isinstance(result, dict):
            result = {
                "query": "",
                "answer": str(result)
            }
        
        # Update state with results
        state["sql_query"] = result.get("query", "")
        state["sql_results"] = result.get("answer", "")
        state["agent_response"] = result["answer"]
        state["messages"].append(AIMessage(content=result["answer"]))
        
        # ENHANCEMENT: Update conversation memory even for SQL queries
        update_sql_context(state, resolved_query, result)
        
        print(f"✅ SQL query completed: {len(result.get('answer', ''))} chars")
        
    except Exception as e:
        error_msg = f"❌ Error processing SQL query: {str(e)}"
        state["sql_results"] = error_msg
        state["agent_response"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
        print(f"SQL query error: {e}")
    
    return state

def resolve_contextual_references(user_input: str, memory: ConversationMemory) -> str:
    """Resolve pronouns and contextual references in SQL queries"""
    
    # Get recent context from conversation history
    recent_context = {}
    if memory.turn_history:
        # Look at last few turns for context
        for turn in reversed(memory.turn_history[-3:]):  # Last 3 turns
            recent_context.update(turn.extracted_entities)
    
    # Also use active context
    recent_context.update(memory.active_context)
    
    user_lower = user_input.lower()
    
    # Handle common pronouns and references
    if any(ref in user_lower for ref in ["them", "those", "it", "that"]):
        print(f"🔗 Resolving contextual reference in: '{user_input}'")
        print(f"🧠 Available context: {recent_context}")
        
        # Build context-aware query
        context_parts = []
        if "brand" in recent_context:
            context_parts.append(f"{recent_context['brand']} brand")
        if "product_type" in recent_context:
            context_parts.append(recent_context["product_type"])
        if "color" in recent_context:
            context_parts.append(recent_context["color"])
            
        if context_parts:
            context_string = " ".join(context_parts)
            
            # Replace pronouns with context
            resolved_query = user_input
            for pronoun in ["them", "those", "it", "that"]:
                resolved_query = resolved_query.replace(pronoun, context_string)
            
            print(f"✅ Resolved query: '{resolved_query}'")
            return resolved_query
    
    # No resolution needed
    return user_input

def update_sql_context(state: ProductSearchState, resolved_query: str, sql_result: dict):
    """Update conversation context after SQL query"""
    
    # Extract any entities from the resolved query
    # This helps maintain context for follow-up questions
    from shopping_assistant.nodes.entity_extractor import entity_extractor_node
    
    # Create a temporary state for entity extraction
    temp_state = state.copy()
    temp_state["user_input"] = resolved_query
    temp_state["has_image"] = False
    temp_state["image_path"] = None
    
    # Extract entities from resolved query
    temp_state = entity_extractor_node(temp_state)
    extracted_entities = temp_state.get("raw_entities", {})
    
    # Update conversation memory
    current_turn = ConversationTurn(
        turn_id=state["turn_count"],
        user_input=state["user_input"],
        extracted_entities=extracted_entities,
        timestamp=datetime.now(),
        search_results=[],  # SQL queries don't have search results
        context_snapshot=extracted_entities.copy()
    )
    
    state["conversation_memory"].add_turn(current_turn)
    
    # Update active context
    state["conversation_memory"].active_context.update(extracted_entities)
    
    print(f"🧵 Updated SQL context: {extracted_entities}")
    print(f"🧠 Active context: {state['conversation_memory'].active_context}")