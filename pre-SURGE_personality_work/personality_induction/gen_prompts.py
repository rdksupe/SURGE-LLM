from openai import OpenAI
import pandas as pd 
import os 
def main():
    df = pd.read_csv('personality_prompts.csv')
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return None

    # Initialize the OpenAI client
    
    for x  in df['prompt']:
        print(x)

        client = OpenAI(api_key=api_key)

        client = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                { "role": "system", "content": "Just return the answer without any explanations of any sort"},
                {"role": "user", "content": x}
            ]
        )

        response = client.choices[0].message.content
        print(response)
        df.loc[df['prompt'] == x, 'prompt'] = response
   
   
        df.to_csv('personality_questions.csv', index=False)


if __name__ == "__main__":  
    main()



