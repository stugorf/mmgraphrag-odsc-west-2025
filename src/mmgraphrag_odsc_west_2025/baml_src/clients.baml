client<llm> ChatGPTAudioPreview {
  provider "openai"
  options {
    model "gpt-4o-audio-preview"
    api_key env.OPENAI_API_KEY
    temperature 0.0
  }
}

client<llm> OllamaGemma3_12b {
  provider "openai-generic"
  options {
    base_url "http://localhost:11434/v1"
    model gemma3:12b
    temperature 0.1
    top_k 64
    top_p 0.95
    min_p 0.0
  }
}

client<llm> OllamaPhi4 {
  provider "openai-generic"
  options {
    base_url "http://localhost:11434/v1"
    model phi4
    temperature 0.0
  }
}

client<llm> OllamaLlama3Vision {
  provider "openai-generic"
  options {
    base_url "http://localhost:11434/v1"
    model llama3.2-vision:11b
    temperature 0.0
  }
}

client<llm> OllamaLLaVA {
  provider "openai-generic"
  options {
    base_url "http://localhost:11434/v1"
    model llava
    temperature 0.0
  }
}

client<llm> OllamaLlama3 {
  provider "openai-generic"
  options {
    base_url "http://localhost:11434/v1"
    model llama3.2:3b
    temperature 0.0
  }
}

retry_policy Constant {
  max_retries 3
  // Strategy is optional
  strategy {
    type constant_delay
    delay_ms 200
  }
}

retry_policy Exponential {
  max_retries 2
  // Strategy is optional
  strategy {
    type exponential_backoff
    delay_ms 300
    mutliplier 1.5
    max_delay_ms 10000
  }
}