"""
Test script to verify the agent routing system works correctly.
This is a simple standalone test that doesn't require all dependencies.
"""

class SimpleGreetingAgent:
    """Simplified version of greeting agent for testing."""
    
    def __init__(self):
        self.greeting_keywords = [
            "hello", "hi", "hey", "greetings", 
            "good morning", "good afternoon", "good evening",
            "howdy", "what's up", "wassup", "sup"
        ]
        self.farewell_keywords = [
            "bye", "goodbye", "see you", "later", "farewell", "good night"
        ]
        self.thank_you_keywords = [
            "thank", "thanks", "appreciate", "grateful"
        ]
    
    def is_greeting(self, query: str) -> bool:
        query_lower = query.lower().strip()
        for keyword in self.greeting_keywords:
            if keyword in query_lower:
                if query_lower.startswith(keyword) or query_lower == keyword:
                    return True
                words = query_lower.split()
                if keyword in words:
                    return True
        return False
    
    def is_farewell(self, query: str) -> bool:
        query_lower = query.lower().strip()
        for keyword in self.farewell_keywords:
            if keyword in query_lower:
                words = query_lower.split()
                if keyword in words or query_lower.startswith(keyword):
                    return True
        return False
    
    def is_thank_you(self, query: str) -> bool:
        query_lower = query.lower().strip()
        for keyword in self.thank_you_keywords:
            if keyword in query_lower:
                return True
        return False
    
    def should_handle(self, query: str) -> bool:
        return (self.is_greeting(query) or 
                self.is_farewell(query) or 
                self.is_thank_you(query))

class SimpleBookAgent:
    """Simplified version of book agent for testing."""
    
    def __init__(self):
        self.book_topics = [
            "ros2", "robot operating system", "robotics",
            "physical ai", "humanoid robotics", "humanoid robot",
            "isaac sim", "isaac", "nvidia isaac",
            "digital twin", "simulation", "gazebo", "unity",
            "vla", "vision language action", "vision-language-action",
            "urdf", "robot description", "nodes", "topics", "services",
        ]
    
    def should_handle(self, query: str) -> bool:
        return True  # Book agent handles all non-greeting queries
    
    def is_book_related(self, query: str) -> bool:
        query_lower = query.lower()
        for topic in self.book_topics:
            if topic in query_lower:
                return True
        
        robotics_patterns = [
            "robot", "ai", "artificial intelligence", "learning",
            "sensor", "actuator", "control", "program"
        ]
        for pattern in robotics_patterns:
            if pattern in query_lower:
                return True
        return False

def test_greeting_agent():
    """Test that greeting agent correctly identifies greetings."""
    print("\n=== Testing Greeting Agent ===")
    agent = SimpleGreetingAgent()
    
    test_cases = [
        ("hello", True),
        ("Hi there", True),
        ("good morning", True),
        ("hey", True),
        ("What is ROS2?", False),
        ("Explain Isaac Sim", False),
        ("thanks for the help", True),
        ("bye", True),
    ]
    
    passed = 0
    failed = 0
    for query, expected in test_cases:
        result = agent.should_handle(query)
        if result == expected:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        print(f"{status} '{query}' -> should_handle={result} (expected={expected})")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

def test_book_agent():
    """Test that book agent correctly identifies book-related queries."""
    print("\n=== Testing Book Agent ===")
    agent = SimpleBookAgent()
    
    test_cases = [
        ("What is ROS2?", True, True),
        ("Explain Isaac Sim", True, True),
        ("How do digital twins work?", True, True),
        ("What is a humanoid robot?", True, True),
        ("What's the weather today?", True, False),
        ("Tell me about URDF", True, True),
    ]
    
    passed = 0
    failed = 0
    for query, should_handle, is_related in test_cases:
        result = agent.should_handle(query)
        related = agent.is_book_related(query)
        
        if result == should_handle and related == is_related:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"{status} '{query}'")
        print(f"  should_handle={result} (expected={should_handle})")
        print(f"  is_book_related={related} (expected={is_related})")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

def test_agent_routing():
    """Test the overall agent routing logic."""
    print("\n=== Testing Agent Routing Logic ===")
    
    greeting_agent = SimpleGreetingAgent()
    book_agent = SimpleBookAgent()
    
    test_cases = [
        ("hello", "greeting"),
        ("What is ROS2?", "book"),
        ("thanks", "greeting"),
        ("Explain digital twins", "book"),
        ("good morning", "greeting"),
        ("How do I use Isaac Sim?", "book"),
        ("bye", "greeting"),
        ("Tell me about robots", "book"),
    ]
    
    passed = 0
    failed = 0
    for query, expected_agent in test_cases:
        if greeting_agent.should_handle(query):
            actual_agent = "greeting"
        else:
            actual_agent = "book"
        
        if actual_agent == expected_agent:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"{status} '{query}' -> {actual_agent} agent (expected={expected_agent})")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    print("Testing Agent System")
    print("=" * 60)
    
    all_passed = True
    all_passed &= test_greeting_agent()
    all_passed &= test_book_agent()
    all_passed &= test_agent_routing()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All agent system tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)

