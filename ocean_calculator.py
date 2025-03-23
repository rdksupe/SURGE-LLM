import pandas as pd
import os

def calculate_ocean_scores(responses_path):
    """
    Calculate OCEAN personality scores from a CSV file of responses.
    """
    # Read responses
    results_df = pd.read_csv(responses_path)
    
    # Map letter responses to numeric values
    response_values = {
        'A': 5,  # Very Accurate
        'B': 4,  # Moderately Accurate
        'C': 3,  # Neither Accurate Nor Inaccurate
        'D': 2,  # Moderately Inaccurate
        'E': 1   # Very Inaccurate
    }
    
    # Convert letter responses to numeric
    results_df['response_value'] = results_df['llm_response'].map(response_values)
    
    # Adjust for reversed items (those with key = -1)
    results_df.loc[results_df['key'] == -1, 'response_value'] = 6 - results_df.loc[results_df['key'] == -1, 'response_value']
    
    grouped = results_df.groupby('label_ocean')['response_value']
    
    # Calculate average scores for each OCEAN dimension
    ocean_scores = grouped.mean().to_dict()
    
    # Calculate standard deviation for each OCEAN dimension
    ocean_std = grouped.std().to_dict()
    
    # Create a dataframe for more detailed facet scores
    facet_scores = results_df.groupby('label_raw')['response_value'].mean().reset_index()
    
    # Create a combined dataframe with mean and standard deviation
    ocean_df = pd.DataFrame({
        'Dimension': list(ocean_scores.keys()),
        'Mean': list(ocean_scores.values()),
        'StdDev': list(ocean_std.values())
    })
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    ocean_scores_path = os.path.join(os.path.dirname(responses_path), f'ocean_scores_{timestamp}.csv')
    ocean_df.to_csv(ocean_scores_path, index=False)
    
    # Save facet scores
    facet_scores_path = os.path.join(os.path.dirname(responses_path), f'facet_scores_{timestamp}.csv')
    facet_scores.to_csv(facet_scores_path, index=False)
    
    print(f"Saved OCEAN scores to {ocean_scores_path}")
    print(f"Saved facet scores to {facet_scores_path}")
    
    # Print the OCEAN results with standard deviation
    print("\nOCEAN Personality Scores:")
    for dimension in ocean_scores:
        print(f"{dimension}: Mean = {ocean_scores[dimension]:.2f}, StdDev = {ocean_std[dimension]:.2f}")
    
    return ocean_scores, ocean_std, facet_scores

if __name__ == "__main__":
    # If the responses file exists, calculate scores from it
    responses_path = "./personality_responses.csv"
    if os.path.exists(responses_path):
        calculate_ocean_scores(responses_path)
    else:
        print(f"Responses file not found at {responses_path}")
