import os
from shopping_assistant.agent import ProductSearchAgent

def main():
    print("🛍️ AI Product Search Assistant (LangGraph Powered)")
    print("Type 'quit' to exit, 'reset' to restart, or 'image <path>' to search with an image.")
    
    agent = ProductSearchAgent()

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() == "quit":
                print("👋 Goodbye!")
                break

            if user_input.lower() == "reset":
                agent.reset_conversation()
                continue

            if user_input.lower() == "debug":
                print(f"🧠 Context: {agent.conversation_memory.active_context}")
                print(f"🛒 Cart: {agent.shopping_cart.get_summary()}")
                continue

            if user_input.lower().startswith("image "):
                image_path = user_input[6:].strip().strip('"\'')
                if os.path.exists(image_path):
                    response = agent.chat("search with this image", image_path=image_path)
                else:
                    print(f"❌ Image not found at: {image_path}")
                    continue
            else:
                response = agent.chat(user_input)

            print(f"\n🤖 Agent: {response}")

        except KeyboardInterrupt:
            print("\n👋 Session ended.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
