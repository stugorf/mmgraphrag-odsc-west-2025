// Dynamic In-Context Learning (dICL) BAML functions
// This file provides functions for generating examples and processing queries
// with dynamic in-context learning using similar examples from a database.

// Type definitions for dICL system
class Example {
    id string @description("Unique identifier for the example")
    input string @description("The input/question for this example")
    output string @description("The output/answer for this example")
}

class ExampleGenerationInput {
    topic string @description("The topic to generate examples about")
    num_examples int @description("Number of examples to generate")
}

class ExampleGenerationOutput {
    examples Example[] @description("List of generated examples")
}

class DICLInput {
    query string @description("The user's query/question")
    examples Example[] @description("Similar examples to use as context")
}

class DICLOutput {
    answer string @description("The answer to the user's query")
    reasoning string @description("The reasoning process used to arrive at the answer")
}

function GenerateExamples(input: ExampleGenerationInput) -> ExampleGenerationOutput {
    client OllamaPhi4
    prompt #"
        You are an expert at creating high-quality question-answer examples for educational purposes.
        
        Topic: {{input.topic}}
        Number of examples needed: {{input.num_examples}}
        
        Create {{input.num_examples}} diverse, educational question-answer pairs about {{input.topic}}.
        Each example should be:
        - Clear and well-structured
        - Educational and informative
        - Cover different aspects of the topic
        - Have a concise but complete answer
        - Be suitable for use as few-shot learning examples
        
        Format each example as:
        Question: [Your question here]
        Answer: [Your detailed answer here]
        
        {{ ctx.output_format }}
    "#
}

function DynamicInContextLearning(input: DICLInput) -> DICLOutput {
    client OllamaPhi4
    prompt #"
        You are an expert assistant that uses dynamic in-context learning to answer questions.
        
        You have been provided with similar examples to help you understand the context and style for answering the user's question.
        
        Similar Examples:
        {{ input.examples }}
        
        User's Question: {{input.query}}
        
        Based on the similar examples provided above, answer the user's question in a similar style and depth.
        Provide both a clear answer and explain your reasoning process.
        
        {{ ctx.output_format }}
    "#
}

// Test cases for the dICL functions
test test_generate_examples_bananas {
    functions [GenerateExamples]
    args {
        input {
            topic "bananas"
            num_examples 3
        }
    }
}

test test_generate_examples_technology {
    functions [GenerateExamples]
    args {
        input {
            topic "machine learning"
            num_examples 2
        }
    }
}

test test_dynamic_in_context_learning {
    functions [DynamicInContextLearning]
    args {
        input {
            query "What are the health benefits of bananas?"
            examples [
                {
                    id "banana_1"
                    input "What nutrients are in bananas?"
                    output "Bananas are rich in potassium, vitamin C, vitamin B6, fiber, and antioxidants. They also contain small amounts of magnesium, folate, and iron."
                }
                {
                    id "banana_2"
                    input "How do bananas help with digestion?"
                    output "Bananas contain dietary fiber, particularly pectin and resistant starch, which help promote healthy digestion, regulate bowel movements, and support gut bacteria."
                }
            ]
        }
    }
}
