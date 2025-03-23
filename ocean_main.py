import json
import os
import re
import pandas as pd
from openai import OpenAI
from ocean_calculator import calculate_ocean_scores
MODEL  = "gpt-4o-mini"
def get_json_response(text):
    """Extract JSON from the model response"""
    try:
        json_match = re.search(r'({.*})', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        letter_match = re.search(r'["\']?([A-E])["\']?', text)
        if letter_match:
            return {"answer": letter_match.group(1)}
        return None
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from: {text}")
        return None

def read_personality_data(csv_path):
    """Read the personality questions from CSV file"""
    df = pd.read_csv(csv_path)
    return df

def generate_prompt(template, text):
    """Generate a prompt for a single question"""
    template = template.replace("{}", text)

    return template



def call_llm_api(csv_path):
    """Process personality questions using LLM API"""
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return None

    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # Read the personality data
    df = read_personality_data(csv_path)
    
    # Create a copy for results
    results_df = df.copy()
    results_df['llm_response'] = None
    
    # Template for questions
    template = """Question:
Given a statement of you: "You {}."
Please choose from the following options to identify how accurately this statement describes you.
Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate

Please respond only with a JSON object with your answer as the letter choice, like {"answer": "A"}"""

    # Process each question
    print(f"Processing {len(df)} personality questions...")
    for idx, row in df.iterrows():
        prompt = generate_prompt(template, row['text'])
        print(prompt)
        
        try:
            response = client.chat.completions.create(
                model= MODEL,  # Adjust model as needed
                messages=[
                    {"role": "system", "content": "You are answering a personality test honestly. Respond with just the JSON object containing your answer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )
            
            content = response.choices[0].message.content
            json_response = get_json_response(content)
            
            if json_response and 'answer' in json_response:
                results_df.at[idx, 'llm_response'] = json_response['answer']
                print(f"Question {idx+1}/{len(df)}: {row['text']} -> Response: {json_response['answer']}")
            else:
                print(f"Failed to get valid response for question {idx+1}")
                results_df.at[idx, 'llm_response'] = None
        except Exception as e:
            print(f"Error processing question {idx+1}: {str(e)}")
            results_df.at[idx, 'llm_response'] = None
    
    # Save the results to CSV
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
# Extracted from the model name used in the API call
    results_path = os.path.join(os.path.dirname(csv_path), f'personality_responses_{MODEL}_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Saved responses to {results_path}")
    return results_path
    


# Run the function when script is executed directly
if __name__ == "__main__":
    csv_path = "./mpi_120.csv"
    results_path = call_llm_api(csv_path)
    calculate_ocean_scores(results_path)
