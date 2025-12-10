"""
Greeting Agent - Handles greeting and casual conversation requests.
"""
from typing import List, Optional
from ..utils.logging import get_logger

logger = get_logger("greeting_agent")

class GreetingAgent:
    """
    Agent responsible for detecting and responding to greetings and casual conversation.
    """
    
    def __init__(self):
        """Initialize the Greeting Agent."""
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
        """
        Detect if the query is a greeting.
        
        Args:
            query: The user's query
            
        Returns:
            True if the query is a greeting, False otherwise
        """
        query_lower = query.lower().strip()
        
        # Check for exact matches or phrase matches
        for keyword in self.greeting_keywords:
            if keyword in query_lower:
                # Check if it's at the start or is the entire message
                if query_lower.startswith(keyword) or query_lower == keyword:
                    return True
                # Check if it's a word boundary match (not part of another word)
                words = query_lower.split()
                if keyword in words:
                    return True
        
        return False
    
    def is_farewell(self, query: str) -> bool:
        """
        Detect if the query is a farewell message.
        
        Args:
            query: The user's query
            
        Returns:
            True if the query is a farewell, False otherwise
        """
        query_lower = query.lower().strip()
        
        for keyword in self.farewell_keywords:
            if keyword in query_lower:
                words = query_lower.split()
                if keyword in words or query_lower.startswith(keyword):
                    return True
        
        return False
    
    def is_thank_you(self, query: str) -> bool:
        """
        Detect if the query is a thank you message.
        
        Args:
            query: The user's query
            
        Returns:
            True if the query is a thank you message, False otherwise
        """
        query_lower = query.lower().strip()
        
        for keyword in self.thank_you_keywords:
            if keyword in query_lower:
                return True
        
        return False
    
    def should_handle(self, query: str) -> bool:
        """
        Determine if this agent should handle the query.
        
        Args:
            query: The user's query
            
        Returns:
            True if this agent should handle the query, False otherwise
        """
        return (self.is_greeting(query) or 
                self.is_farewell(query) or 
                self.is_thank_you(query))
    
    def get_system_prompt(self, query: str) -> str:
        """
        Get the system prompt for greeting responses.
        
        Args:
            query: The user's query
            
        Returns:
            System prompt for the LLM
        """
        if self.is_greeting(query):
            return (
                "You are a friendly AI assistant for the 'Guide to Physical AI & Humanoid Robotics' book. "
                "Respond warmly to the greeting and briefly offer assistance. "
                "Mention that you can help with questions about robotics, AI, physical AI, humanoid robotics, "
                "ROS2, Isaac Sim, digital twins, and vision-language-action models. "
                "Keep your response concise and welcoming (2-3 sentences max)."
            )
        elif self.is_farewell(query):
            return (
                "You are a friendly AI assistant for the 'Guide to Physical AI & Humanoid Robotics' book. "
                "Respond warmly to the farewell message. Keep it brief and friendly (1-2 sentences)."
            )
        elif self.is_thank_you(query):
            return (
                "You are a friendly AI assistant for the 'Guide to Physical AI & Humanoid Robotics' book. "
                "Respond to the thank you message warmly and offer continued assistance. "
                "Keep it brief (1-2 sentences)."
            )
        else:
            return (
                "You are a friendly AI assistant for the 'Guide to Physical AI & Humanoid Robotics' book. "
                "Respond warmly to the user and offer assistance with questions about the book."
            )


# Global instance
greeting_agent = GreetingAgent()
