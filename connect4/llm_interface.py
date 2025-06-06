import os
import requests
import json
import time
from typing import Dict, Any, Optional, Tuple, List

class LLMInterface:
    """Interface for communicating with LLM APIs"""
    
    def __init__(self, model="gemma3:latest"):
        """
        Initialize the LLM interface for Ollama
        
        Args:
            model: Model to use ('gemma3:latest', 'qwen3:latest', or 'deepseek-r1:8b')
        """
        self.model = model
        self.supported_models = ["gemma3:latest", "qwen3:latest", "deepseek-r1:8b"]
        self.timeout = 90  # Timeout in seconds (90 seconds)
        self.conversation_history = []
        
        # All models can use thinking tags in their responses
        self.supports_thinking = True
        
        if model not in self.supported_models:
            print(f"Warning: Model {model} not in supported models list. Using gemma3:latest instead.")
            self.model = "gemma3:latest"
    
    def get_response(self, prompt: str) -> Tuple[str, Dict]:
        """
        Get a response from Ollama using Chat API
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Tuple of (response_text, full_response_data)
        """
        start_time = time.time()
        print(f"Requesting response from {self.model} at {start_time}")

        # Add the new prompt to conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        
        try:
            # Prepare request data for chat API
            request_data = {
                "model": self.model,
                "messages": self.conversation_history,
                "stream": False
            }
            
            # Add options parameter for models that support thinking
            # The correct format is "options" at the top level with "temperature" etc.
            if self.supports_thinking:
                request_data["options"] = {
                    "temperature": 0.7,
                    "num_predict": -1  # Set to -1 for unlimited token generation
                }
            
            # Call Ollama Chat API with timeout
            response = requests.post(
                "http://localhost:11434/api/chat",
                json=request_data,
                timeout=self.timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"Response received in {duration:.2f} seconds")
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Log the full response from Ollama API
                print("Full Ollama API response:")
                print(json.dumps(response_data, indent=2))
                
                # Extract response content
                response_content = response_data.get("message", {}).get("content", "")
                
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": response_content})
                
                # Extract thinking process if available (from <think> tags in response)
                thinking = ""
                response_content_str = response_content
                if "<think>" in response_content_str and "</think>" in response_content_str:
                    import re
                    think_match = re.search(r"<think>([\s\S]*?)</think>", response_content_str)
                    if think_match:
                        thinking = think_match.group(1).strip()
                        # Remove thinking tags from the response content
                        response_content = response_content_str.replace(think_match.group(0), "").strip()
                
                return response_content, {
                    "response": response_content,
                    "thinking": thinking,
                    "duration": duration,
                    "model": self.model,
                    "status": "success"
                }
            else:
                print(f"Error from Ollama API: {response.text}")
                # Provide a fallback response for game to continue
                fallback = self._get_fallback_response()
                return fallback, {
                    "error": response.text,
                    "duration": duration,
                    "model": self.model,
                    "status": "error",
                    "status_code": response.status_code,
                    "fallback": True
                }
        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            print(f"Timeout after {duration:.2f} seconds when calling Ollama API")
            # Provide a fallback response for game to continue
            fallback = self._get_fallback_response()
            return fallback, {
                "error": "Request timed out",
                "duration": duration,
                "model": self.model,
                "status": "timeout",
                "fallback": True
            }
        except Exception as e:
            duration = time.time() - start_time
            print(f"Exception when calling Ollama API: {e}")
            # Provide a fallback response for game to continue
            fallback = self._get_fallback_response()
            return fallback, {
                "error": str(e),
                "duration": duration,
                "model": self.model,
                "status": "exception",
                "fallback": True
            }
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        print(f"Conversation history reset for {self.model}")
    
    def _get_fallback_response(self) -> str:
        """Get a fallback response when the LLM fails"""
        # For Connect 4, we need a column number (0-6)
        import random
        return str(random.randint(0, 6))
    
    def get_available_models(self):
        """Get list of available models from Ollama"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                return self.supported_models
        except:
            return self.supported_models