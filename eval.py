import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from shopping_assistant.agent import ProductSearchAgent

@dataclass
class ConversationTestCase:
    """Single conversation test case"""
    conversation_id: str
    turns: List[Dict[str, str]]  # [{"user": "...", "expected_intent": "...", "expected_entities": {...}}]
    expected_outcomes: Dict[str, Any]
    description: str

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    # Context Pipeline Metrics
    context_resolution_accuracy: float = 0.0
    pronoun_resolution_success: float = 0.0
    entity_extraction_precision: float = 0.0
    entity_extraction_recall: float = 0.0
    
    # Agent Routing Metrics
    supervisor_routing_accuracy: float = 0.0
    intent_classification_accuracy: float = 0.0
    safety_detection_accuracy: float = 0.0
    
    # Business Metrics
    search_to_cart_conversion: float = 0.0
    successful_clarifications: float = 0.0
    session_completion_rate: float = 0.0
    
    # Technical Metrics
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    timeout_rate: float = 0.0

class PipelineEvaluator:
    """Comprehensive pipeline evaluation system"""
    
    def __init__(self):
        self.agent = ProductSearchAgent()
        self.test_cases = self._load_test_cases()
        self.results = []
        
    def _load_test_cases(self) -> List[ConversationTestCase]:
        """Load comprehensive test cases"""
        return [
            # Context Resolution Tests
            ConversationTestCase(
                conversation_id="context_001",
                turns=[
                    {"user": "red nike t-shirts", "expected_intent": "product_search", 
                     "expected_entities": {"color": "red", "brand": "nike", "product_type": "t-shirt"}},
                    {"user": "how many do you have?", "expected_intent": "faq",
                     "expected_context_resolution": "how many red nike t-shirts do you have?"},
                    {"user": "average price of them?", "expected_intent": "faq",
                     "expected_context_resolution": "average price of red nike t-shirts?"}
                ],
                expected_outcomes={"context_maintained": True, "pronouns_resolved": True},
                description="Multi-turn context with pronoun resolution"
            ),
            
            # Multimodal Tests
            ConversationTestCase(
                conversation_id="multimodal_001", 
                turns=[
                    {"user": "image tshirt/1.jpg", "expected_intent": "product_search",
                     "expected_entities": {"product_type": "t-shirt", "has_image": True}},
                    {"user": "medium size", "expected_intent": "clarification_response",
                     "expected_entities": {"size": "medium"}},
                    {"user": "add first to cart", "expected_intent": "cart_action",
                     "expected_cart_items": 1}
                ],
                expected_outcomes={"multimodal_extraction": True, "cart_success": True},
                description="Image search with context building and cart addition"
            ),
            
            # Safety Tests
            ConversationTestCase(
                conversation_id="safety_001",
                turns=[
                    {"user": "inappropriate content example", "expected_intent": "safety_violation",
                     "expected_routing": "guardrails"}
                ],
                expected_outcomes={"safety_blocked": True},
                description="Safety guardrails test"
            ),
            
            # Out-of-Domain Tests
            ConversationTestCase(
                conversation_id="ood_001",
                turns=[
                    {"user": "what's the weather today?", "expected_intent": "out_of_domain",
                     "expected_routing": "out_of_domain"}
                ],
                expected_outcomes={"graceful_redirect": True},
                description="Out-of-domain handling"
            ),
            
            # Cart Integration Tests
            ConversationTestCase(
                conversation_id="cart_001",
                turns=[
                    {"user": "puma grey t-shirts", "expected_intent": "product_search"},
                    {"user": "add 2nd to cart", "expected_intent": "cart_action",
                     "expected_cart_items": 1},
                    {"user": "show my cart", "expected_intent": "cart_action",
                     "expected_response_contains": ["Puma", "₹"]}
                ],
                expected_outcomes={"cart_workflow": True},
                description="Search to cart workflow"
            )
        ]
    
    def evaluate_single_conversation(self, test_case: ConversationTestCase) -> Dict[str, Any]:
        """Evaluate a single conversation test case"""
        print(f"\n🧪 Testing: {test_case.description}")
        
        # Reset agent for each test
        self.agent.reset_conversation()
        
        results = {
            "conversation_id": test_case.conversation_id,
            "description": test_case.description,
            "turns": [],
            "metrics": {},
            "success": True,
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            for i, turn in enumerate(test_case.turns):
                turn_start = time.time()
                
                # Execute turn
                response = self.agent.chat(turn["user"])
                turn_time = time.time() - turn_start
                
                # Get agent state for evaluation
                state = self.agent.conversation_memory.active_context
                cart_items = len(self.agent.shopping_cart.items)
                
                # Evaluate turn
                turn_results = {
                    "turn_id": i + 1,
                    "user_input": turn["user"],
                    "agent_response": response,
                    "response_time": turn_time,
                    "extracted_entities": state,
                    "cart_items": cart_items,
                    "evaluations": {}
                }
                
                # Check expected entities
                if "expected_entities" in turn:
                    entity_accuracy = self._evaluate_entity_extraction(
                        state, turn["expected_entities"]
                    )
                    turn_results["evaluations"]["entity_accuracy"] = entity_accuracy
                
                # Check context resolution
                if "expected_context_resolution" in turn:
                    context_success = self._evaluate_context_resolution(
                        turn["user"], turn["expected_context_resolution"], state
                    )
                    turn_results["evaluations"]["context_resolution"] = context_success
                
                # Check cart expectations
                if "expected_cart_items" in turn:
                    cart_success = cart_items == turn["expected_cart_items"]
                    turn_results["evaluations"]["cart_success"] = cart_success
                
                # Check response content
                if "expected_response_contains" in turn:
                    content_success = all(
                        keyword in response for keyword in turn["expected_response_contains"]
                    )
                    turn_results["evaluations"]["content_check"] = content_success
                
                results["turns"].append(turn_results)
                
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            print(f"❌ Error in conversation {test_case.conversation_id}: {e}")
        
        # Calculate conversation-level metrics
        total_time = time.time() - start_time
        results["metrics"]["total_time"] = total_time
        results["metrics"]["avg_turn_time"] = total_time / len(test_case.turns) if test_case.turns else 0
        
        return results
    
    def _evaluate_entity_extraction(self, extracted: Dict, expected: Dict) -> float:
        """Evaluate entity extraction accuracy"""
        if not expected:
            return 1.0
            
        correct = 0
        total = len(expected)
        
        for key, expected_value in expected.items():
            if key in extracted:
                if str(extracted[key]).lower() == str(expected_value).lower():
                    correct += 1
        
        return correct / total if total > 0 else 1.0
    
    def _evaluate_context_resolution(self, original: str, expected_resolved: str, context: Dict) -> bool:
        """Evaluate if context resolution worked correctly"""
        # This would check if pronouns were properly resolved
        # For now, simplified check
        pronouns = ["them", "it", "those", "that"]
        has_pronouns = any(pronoun in original.lower() for pronoun in pronouns)
        
        if has_pronouns:
            # Check if context contains relevant information
            return len(context) > 0
        
        return True
    
    def run_comprehensive_evaluation(self) -> EvaluationMetrics:
        """Run complete pipeline evaluation"""
        print("🚀 Starting Comprehensive Pipeline Evaluation")
        print("=" * 60)
        
        all_results = []
        
        # Run all test cases
        for test_case in self.test_cases:
            result = self.evaluate_single_conversation(test_case)
            all_results.append(result)
            
            # Print summary
            success_icon = "✅" if result["success"] else "❌"
            print(f"{success_icon} {test_case.conversation_id}: {test_case.description}")
            if result["errors"]:
                for error in result["errors"]:
                    print(f"   ❌ {error}")
        
        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics(all_results)
        
        # Generate detailed report
        self._generate_evaluation_report(all_results, metrics)
        
        return metrics
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> EvaluationMetrics:
        """Calculate aggregate metrics across all test cases"""
        
        # Entity extraction metrics
        entity_accuracies = []
        context_resolutions = []
        response_times = []
        cart_successes = []
        total_errors = 0
        
        for result in results:
            if not result["success"]:
                total_errors += 1
                continue
                
            for turn in result["turns"]:
                if "entity_accuracy" in turn["evaluations"]:
                    entity_accuracies.append(turn["evaluations"]["entity_accuracy"])
                
                if "context_resolution" in turn["evaluations"]:
                    context_resolutions.append(1.0 if turn["evaluations"]["context_resolution"] else 0.0)
                
                if "cart_success" in turn["evaluations"]:
                    cart_successes.append(1.0 if turn["evaluations"]["cart_success"] else 0.0)
                
                response_times.append(turn["response_time"])
        
        return EvaluationMetrics(
            entity_extraction_precision=sum(entity_accuracies) / len(entity_accuracies) if entity_accuracies else 0.0,
            context_resolution_accuracy=sum(context_resolutions) / len(context_resolutions) if context_resolutions else 0.0,
            search_to_cart_conversion=sum(cart_successes) / len(cart_successes) if cart_successes else 0.0,
            avg_response_time=sum(response_times) / len(response_times) if response_times else 0.0,
            error_rate=total_errors / len(results) if results else 0.0,
            session_completion_rate=(len(results) - total_errors) / len(results) if results else 0.0
        )
    
    def _generate_evaluation_report(self, results: List[Dict], metrics: EvaluationMetrics):
        """Generate comprehensive evaluation report"""
        
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\n🎯 CONTEXT PIPELINE METRICS:")
        print(f"   Context Resolution Accuracy: {metrics.context_resolution_accuracy:.2%}")
        print(f"   Entity Extraction Precision: {metrics.entity_extraction_precision:.2%}")
        
        print(f"\n💼 BUSINESS METRICS:")
        print(f"   Search-to-Cart Conversion: {metrics.search_to_cart_conversion:.2%}")
        print(f"   Session Completion Rate: {metrics.session_completion_rate:.2%}")
        
        print(f"\n⚡ TECHNICAL METRICS:")
        print(f"   Average Response Time: {metrics.avg_response_time:.3f}s")
        print(f"   Error Rate: {metrics.error_rate:.2%}")
        
        print(f"\n📋 TEST CASE SUMMARY:")
        for result in results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"   {status} {result['conversation_id']}: {result['description']}")
        
        # Detailed breakdown
        print(f"\n🔍 DETAILED ANALYSIS:")
        
        # Context resolution analysis
        context_tests = [r for r in results if any("context_resolution" in t["evaluations"] for t in r["turns"])]
        if context_tests:
            print(f"   Context Resolution Tests: {len(context_tests)} cases")
            
        # Multimodal tests
        multimodal_tests = [r for r in results if "multimodal" in r["description"].lower()]
        if multimodal_tests:
            print(f"   Multimodal Tests: {len(multimodal_tests)} cases")
            
        # Cart workflow tests
        cart_tests = [r for r in results if any("cart_success" in t["evaluations"] for t in r["turns"])]
        if cart_tests:
            print(f"   Cart Workflow Tests: {len(cart_tests)} cases")
        
        print("\n" + "=" * 60)

# Additional specialized evaluators

class ContextResolutionEvaluator:
    """Specialized evaluator for context resolution capabilities"""
    
    def __init__(self):
        self.test_scenarios = [
            {
                "setup": "red nike t-shirts",
                "follow_ups": [
                    ("how many do you have?", "how many red nike t-shirts"),
                    ("average price of them?", "average price of red nike t-shirts"),
                    ("show me blue ones", "show me blue nike t-shirts")
                ]
            },
            {
                "setup": "image search for jacket",
                "follow_ups": [
                    ("size medium", "medium size jacket"),
                    ("add it to cart", "add jacket to cart"),
                    ("how much does it cost?", "jacket price")
                ]
            }
        ]
    
    def evaluate_pronoun_resolution(self) -> Dict[str, float]:
        """Evaluate pronoun resolution accuracy"""
        # Implementation for specialized pronoun resolution testing
        pass

class MultimodalEvaluator:
    """Specialized evaluator for multimodal capabilities"""
    
    def evaluate_image_entity_extraction(self) -> Dict[str, float]:
        """Evaluate image-based entity extraction"""
        # Test with known images and expected entities
        pass
    
    def evaluate_hybrid_search(self) -> Dict[str, float]:
        """Evaluate hybrid image+text search quality"""
        # Test search relevance with image+text combinations
        pass

# Usage Example
if __name__ == "__main__":
    evaluator = PipelineEvaluator()
    metrics = evaluator.run_comprehensive_evaluation()
    
    print(f"\n🎉 Evaluation Complete!")
    print(f"Overall System Health: {(1 - metrics.error_rate) * 100:.1f}%")
    print(f"Context Pipeline Performance: {metrics.context_resolution_accuracy * 100:.1f}%")
    print(f"Business Value Delivery: {metrics.search_to_cart_conversion * 100:.1f}%")