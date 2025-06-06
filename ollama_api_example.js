// Example of making an API call to Ollama with num_predict set to -1
// This configures the LLM to generate with unlimited token prediction

// Generate completion example
async function generateCompletion() {
  const response = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'llama3.2',
      prompt: 'Write a short story about a robot learning to paint',
      options: {
        num_predict: -1,  // Set to -1 for unlimited token generation
        temperature: 0.7,
        top_p: 0.9
      }
    }),
  });
  
  return response.json();
}

// Chat completion example
async function chatCompletion() {
  const response = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'llama3.2',
      messages: [
        {
          role: 'user',
          content: 'Write a short story about a robot learning to paint'
        }
      ],
      options: {
        num_predict: -1,  // Set to -1 for unlimited token generation
        temperature: 0.7,
        top_p: 0.9
      }
    }),
  });
  
  return response.json();
}

// Example with curl (for command line usage)
/*
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Write a short story about a robot learning to paint",
  "options": {
    "num_predict": -1,
    "temperature": 0.7,
    "top_p": 0.9
  }
}'
*/