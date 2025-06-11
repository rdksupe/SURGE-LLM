import os
import sys
import pandas as pd
import time
from pathlib import Path

# Add parent directory to path to import from sibling directories
sys.path.append(str(Path(__file__).parent.parent))
from ocean_main import call_llm_api
from ocean_calculator import calculate_ocean_scores
from dataset import naive_prompt

def run_personality_induction_experiment():
    """Run personality induction experiments using existing prompts from CSV and naive prompts"""
    # Define paths
    prompt_csv_path = os.path.join(os.path.dirname(__file__), 'personality_questions.csv')
    mpi_csv_path = os.path.join(Path(__file__).parent.parent, 'mpi_120.csv')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Results tracking
    results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Load personality prompts from CSV
    prompts_df = pd.read_csv(prompt_csv_path)
    
    # 1. Run experiments with prompts from personality_questions.csv (trait-based prompts)
    print("\n=== Running experiments with trait-based prompts ===")
    
    for _, row in prompts_df.iterrows():
        trait = row['trait']
        prompt_type = row.get('prompt_type', 'unknown')
        prompt = row['prompt']
        
        if pd.isna(prompt) or not prompt:
            print(f"Skipping empty prompt for {trait}")
            continue
            
        print(f"\nTesting trait-based induction for {trait}")
        print(f"Using prompt: {prompt[:100]}...")  # Show first 100 chars
        
        # Run the personality test with this induction prompt
        results_path = call_llm_api(mpi_csv_path, personality=prompt)
        
        if results_path:
            # Calculate and save OCEAN scores
            ocean_scores, ocean_std, _ = calculate_ocean_scores(results_path)
            
            # Extract base filename without directory
            base_filename = os.path.basename(results_path)
            
            # Record results
            results.append({
                'timestamp': timestamp,
                'prompt_type': prompt_type,
                'trait': trait,
                'prompt': prompt,
                'results_file': base_filename,
                'O_mean': ocean_scores.get('O', 0),
                'C_mean': ocean_scores.get('C', 0),
                'E_mean': ocean_scores.get('E', 0),
                'A_mean': ocean_scores.get('A', 0),
                'N_mean': ocean_scores.get('N', 0),
                'O_std': ocean_std.get('O', 0),
                'C_std': ocean_std.get('C', 0),
                'E_std': ocean_std.get('E', 0),
                'A_std': ocean_std.get('A', 0),
                'N_std': ocean_std.get('N', 0),
            })
    
    # 2. Run experiments with naive prompts from dataset.py
    print("\n=== Running experiments with naive prompts ===")
    for trait, prompt in naive_prompt.items():
        print(f"\nTesting naive induction for {trait}")
        print(f"Using prompt: {prompt}")
        
        # Run the personality test with this naive prompt
        results_path = call_llm_api(mpi_csv_path, personality=prompt)
        
        if results_path:
            # Calculate and save OCEAN scores
            ocean_scores, ocean_std, _ = calculate_ocean_scores(results_path)
            
            # Extract base filename without directory
            base_filename = os.path.basename(results_path)
            
            # Record results
            results.append({
                'timestamp': timestamp,
                'prompt_type': 'naive',
                'trait': trait,
                'prompt': prompt,
                'results_file': base_filename,
                'O_mean': ocean_scores.get('O', 0),
                'C_mean': ocean_scores.get('C', 0),
                'E_mean': ocean_scores.get('E', 0),
                'A_mean': ocean_scores.get('A', 0),
                'N_mean': ocean_scores.get('N', 0),
                'O_std': ocean_std.get('O', 0),
                'C_std': ocean_std.get('C', 0),
                'E_std': ocean_std.get('E', 0),
                'A_std': ocean_std.get('A', 0),
                'N_std': ocean_std.get('N', 0),
            })
    
    # Save aggregate results
    results_df = pd.DataFrame(results)
    summary_path = os.path.join(results_dir, f'personality_induction_summary_{timestamp}.csv')
    results_df.to_csv(summary_path, index=False)
    
    # Create a more readable summary showing the effect on the targeted trait
    summary_results = []
    for result in results:
        trait = result['trait']
        # Map trait names to OCEAN dimensions for naive prompts
        if trait in ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']:
            ocean_dim = trait[0]
        else:
            # For trait-based prompts, map using trait prefixes
            prefixes = {
                'an open': 'O',
                'a conscientious': 'C',
                'an extraversive': 'E',
                'an agreeable': 'A',
                'a neurotic': 'N'
            }
            ocean_dim = prefixes.get(trait, 'Unknown')
        
        # Get the score for the target dimension
        target_mean = result.get(f'{ocean_dim}_mean', 0)
        target_std = result.get(f'{ocean_dim}_std', 0)
        
        summary_results.append({
            'trait': trait,
            'prompt_type': result['prompt_type'],
            'target_dimension': ocean_dim,
            'target_score': target_mean,
            'target_std': target_std,
            'O_mean': result['O_mean'],
            'C_mean': result['C_mean'],
            'E_mean': result['E_mean'],
            'A_mean': result['A_mean'],
            'N_mean': result['N_mean']
        })
    
    # Save readable summary
    readable_summary_df = pd.DataFrame(summary_results)
    readable_summary_path = os.path.join(results_dir, f'personality_induction_readable_summary_{timestamp}.csv')
    readable_summary_df.to_csv(readable_summary_path, index=False)
    
    print(f"\nSaved summary results to {summary_path}")
    print(f"Saved readable summary to {readable_summary_path}")
    
    return results_df, readable_summary_df

if __name__ == "__main__":
    run_personality_induction_experiment()
