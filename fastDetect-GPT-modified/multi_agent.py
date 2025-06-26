import os
import json
import numpy as np
from openai import OpenAI
import textstat
import re
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd

# --- Configuration ---

# The powerful LLM that will act as the "brain" for all our observer agents.
AGENT_MODEL = "gemma-3-4b-it-qat"

# Allow specifying a custom base URL for the OpenAI client
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)

# --- LLM-Powered Agent Base Class ---

class LLMAgent:
    """A base class for an agent powered by a Large Language Model."""
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.name = self.__class__.__name__
        self.system_prompt = "You are a helpful assistant."
        # Flag to determine if we're using a local model (which might have limitations)
        self.is_local_model = OPENAI_BASE_URL is not None and "localhost" in OPENAI_BASE_URL

    def _get_llm_response(self, user_prompt: str, json_mode: bool = True) -> str:
        """Helper to get a response from the configured LLM."""
        try:
            # For local models, we'll skip the response_format parameter
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
            }
            
            # Only add response_format for non-local models and when json_mode is True
            if json_mode and not self.is_local_model:
                kwargs["response_format"] = {"type": "json_object"}
            
            # Adjust the user prompt to request JSON format for local models
            if json_mode and self.is_local_model:
                # Enhance the user prompt to specify JSON output format
                user_prompt_json = f"{user_prompt}\n\nRespond ONLY with valid JSON. Do not include any explanatory text outside the JSON structure."
                kwargs["messages"][1]["content"] = user_prompt_json

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            error_message = f"API call failed for {self.name}: {e}"
            print(error_message)
            return json.dumps({"error": error_message})

    def analyze(self, text: str) -> dict:
        """Triggers the agent's analysis by querying the LLM."""
        print(f"  ({self.name}) Starting analysis...")
        user_prompt = f"Please analyze the following text:\n\n---\n\n{text}"
        response_str = self._get_llm_response(user_prompt)
        try:
            result = json.loads(response_str)
            print(f"  ({self.name}) Analysis complete.")
            return result
        except json.JSONDecodeError:
            print(f"  ({self.name}) Failed to decode response as JSON.")
            print(f"  ({self.name}) Raw response: {response_str[:100]}...")
            return {"error": "Failed to decode LLM response as JSON.", "raw_response": response_str}

# --- Specialized Observer Agents (as LLM Personas) ---

class ProbabilityObserver(LLMAgent):
    """An LLM agent persona focused on statistical analysis of text."""
    def __init__(self, client: OpenAI, model: str):
        super().__init__(client, model)
        self.system_prompt = """
        You are a specialist in computational linguistics with expertise in statistical analysis of text generation patterns.
        
        TASK: Analyze the provided text to determine if it exhibits the statistical characteristics of HUMAN-WRITTEN text.
        
        FOCUS AREAS:
        1. PERPLEXITY ANALYSIS: Evaluate whether the text shows natural perplexity variations typical of human writing.
           - Human text: Often has moderate to high perplexity with natural variation and occasional surprising word choices
           - AI text: Often has low perplexity with highly predictable word sequences and fewer creative leaps
        
        2. PROBABILITY FLOW ANALYSIS: Assess how token probabilities vary throughout the text.
           - Human text: Shows "bursty" patterns with irregular flow - some parts highly predictable, others surprising
           - AI text: Shows smoother probability distribution without major surprises or creative leaps
        
        OUTPUT REQUIREMENTS:
        Respond with a JSON object containing exactly these three keys:
        - "perplexity_analysis": Detailed explanation (1-3 sentences) of perplexity observations
        - "flow_analysis": Detailed explanation (1-3 sentences) of probability flow patterns
        - "human_likelihood_score": A single float between 0.0 (definitely AI-generated) and 1.0 (definitely human-written)
        
        BASE YOUR SCORE ON THESE INDICATORS:
        - Natural irregularities in sentence structures (suggests human)
        - Occasional grammatical imperfections (suggests human)
        - Unpredictable or creative word choices (suggests human)
        - Excessively uniform predictability (suggests AI)
        - Suspiciously perfect grammar throughout (suggests AI)
        - Overly consistent sentence patterns (suggests AI)
        """

class RewritabilityObserver(LLMAgent):
    """An LLM agent persona that measures rewritability."""
    def __init__(self, client: OpenAI, model: str):
        super().__init__(client, model)
        self.system_prompt = """
        You are a language improvement specialist with deep expertise in text refinement and rewriting.
        
        TASK: When provided with text, your job is to rewrite it to improve clarity and flow while preserving all meaning.
        
        IMPORTANT GUIDELINES:
        - Maintain the exact same information content and key points
        - Improve clarity, coherence, and readability
        - Fix awkward phrasing and grammatical issues
        - Do NOT add new ideas, examples, or information
        - Do NOT explain or comment on the text
        - Provide ONLY the rewritten text in your response, nothing else
        
        The purpose of this exercise is to determine how much the text changes during rewriting, which helps identify AI-generated content.
        Human-written text typically requires more edits for optimization, while AI text often needs minimal changes as it's already optimized.
        """

    def _levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2): return self._levenshtein_distance(s2, s1)
        if len(s2) == 0: return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def analyze(self, text: str) -> dict:
        if not text: return {"error": "Input text is empty."}
        print(f"  ({self.name}) Requesting rewrite...")
        rewrite_prompt = f"Please refine and rewrite the following text to improve its clarity and flow. Do not add any new ideas or commentary. Just provide the rewritten text.\n\n---\n\n{text}"
        rewritten_text = self._get_llm_response(rewrite_prompt, json_mode=False)
        if "error" in rewritten_text:
            return {"error": rewritten_text}
        
        print(f"  ({self.name}) Rewrite received.")
        distance = self._levenshtein_distance(text, rewritten_text)
        max_len = max(len(text), len(rewritten_text))
        similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
        
        # New interpretation - higher similarity now suggests AI, lower similarity suggests human
        human_likelihood = 1.0 - similarity
        analysis = "A low similarity score (<0.7) suggests the text needed significant refinements, which is typical of human-written content."
        return {
            "rewrite_similarity_score": round(similarity, 4), 
            "human_likelihood_score": round(human_likelihood, 4),
            "analysis": analysis
        }

class SyntacticObserver(LLMAgent):
    """An LLM agent persona that analyzes stylistic and syntactic features."""
    def __init__(self, client: OpenAI, model: str):
        super().__init__(client, model)
        self.system_prompt = """
        You are a linguistic analyst specializing in stylometry and syntactic pattern detection in text.
        
        TASK: Analyze the provided text for syntactic and stylistic patterns that indicate HUMAN authorship.
        
        FOCUS AREAS:
        1. STRUCTURAL AUTHENTICITY: Examine sentence length, structure variety, and paragraph organization.
           - Human text: Often shows natural variation in sentence length and structure, with occasional awkward constructions
           - AI text: Often has highly consistent, "perfect" sentence structures with predictable variation patterns
        
        2. LEXICAL HUMANITY: Assess vocabulary usage, specificity, and idiosyncrasy.
           - Human text: Usually has uneven vocabulary distribution with domain-specific terms, personal preferences, and occasional repetition
           - AI text: Often shows unnaturally balanced vocabulary distribution and consistently "optimal" word choices
        
        OUTPUT REQUIREMENTS:
        Respond with a JSON object containing exactly these three keys:
        - "structure_analysis": Detailed explanation (2-3 sentences) of structural patterns
        - "lexical_analysis": Detailed explanation (2-3 sentences) of vocabulary characteristics
        - "human_likelihood_score": A single float between 0.0 (definitely AI-generated) and 1.0 (definitely human-written)
        
        BASE YOUR SCORE ON THESE INDICATORS:
        - Natural redundancies or awkward phrasings (suggests human)
        - Idiosyncratic or unique stylistic choices (suggests human)
        - Inconsistent levels of formality or technicality (suggests human)
        - Personal anecdotes or subjective perspectives (suggests human)
        - Unnaturally consistent sentence lengths (suggests AI)
        - Too-perfect paragraph transitions (suggests AI)
        - Suspiciously optimal vocabulary distribution (suggests AI)
        """

# --- The Judge Agent ---

class JudgeAgent(LLMAgent):
    """
    An LLM agent that receives reports from all observers and makes a final,
    reasoned judgment using zero-shot capabilities.
    """
    def __init__(self, client: OpenAI, model: str):
        super().__init__(client, model)
        self.name = "JudgeAgent"
        self.system_prompt = """
        You are the Lead Judge in a panel of text analysis experts with specialized knowledge in computational linguistics and human writing patterns.
        
        TASK: Your critical responsibility is to analyze reports from specialized observer agents and provide a SINGLE FINAL VERDICT on whether a text is HUMAN-WRITTEN or AI-generated.
        
        INPUT REPORTS:
        You will receive analysis reports from three expert observer agents:
        
        1. PROBABILITY OBSERVER: Analyzes statistical predictability patterns in text.
           - Key metric: human_likelihood_score (0.0-1.0, higher suggests HUMAN)
           - Focuses on: perplexity and probability flow patterns
        
        2. REWRITABILITY OBSERVER: Measures how much the text changes when professionally rewritten.
           - Key metric: human_likelihood_score (0.0-1.0, higher suggests HUMAN)
           - Rationale: Human text typically requires more editing to optimize than AI text
        
        3. SYNTACTIC OBSERVER: Examines stylistic consistency and vocabulary patterns.
           - Key metric: human_likelihood_score (0.0-1.0, higher suggests HUMAN)
           - Focuses on: structural authenticity and lexical humanity
        
        DECISION FRAMEWORK:
        1. Weight the ProbabilityObserver's score most heavily (40%)
        2. Weight the SyntacticObserver's score significantly (35%) 
        3. Weight the RewritabilityObserver's score moderately (25%)
        4. Consider all explanations provided by each observer
        
        OUTPUT REQUIREMENTS:
        You MUST provide a JSON response with EXACTLY these fields:
        1. "final_prediction": MUST be ONLY either "Human-Written" or "AI-Generated"
        2. "human_confidence_score": A SINGLE FLOAT between 0.0 and 1.0 representing your confidence that the text is human-written
           - Score of 0.0 means definitely AI-generated
           - Score of 1.0 means definitely human-written
        3. "detailed_reasoning": Brief explanation of your verdict referencing specific evidence from the observer reports
        
        IMPORTANT: Your "human_confidence_score" is critical for evaluation metrics and must be a single floating-point number between 0.0 and 1.0.
        """

    def decide(self, reports: dict) -> dict:
        """Formats the reports and asks the LLM Judge for a final decision."""
        print(f"  ({self.name}) Making final decision based on observer reports...")
        reports_json_str = json.dumps(reports, indent=2)
        user_prompt = f"Here are the reports from the observer agents. Please provide your final judgment on whether the text is HUMAN-WRITTEN or AI-generated.\n\n---\n\n{reports_json_str}"
        response_str = self._get_llm_response(user_prompt)
        
        try:
            # First try standard JSON parsing
            result = json.loads(response_str)
            print(f"  ({self.name}) Decision complete.")
            
            # Convert to standard format for the rest of the code
            if "human_confidence_score" in result and "final_confidence_score" not in result:
                result["final_confidence_score"] = 1.0 - result["human_confidence_score"]
            
            return result
        except json.JSONDecodeError as e:
            print(f"  ({self.name}) Standard JSON parsing failed: {e}")
            
            # Try to extract JSON from markdown code blocks if present
            if "```json" in response_str:
                try:
                    # Extract content between ```json and ``` markers
                    json_content = response_str.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_content)
                    print(f"  ({self.name}) Successfully extracted JSON from markdown code block.")
                    
                    # Convert to standard format for the rest of the code
                    if "human_confidence_score" in result and "final_confidence_score" not in result:
                        result["final_confidence_score"] = 1.0 - result["human_confidence_score"]
                    
                    return result
                except Exception as e2:
                    print(f"  ({self.name}) Failed to extract JSON from markdown: {e2}")
            
            # If all else fails, return error
            print(f"  ({self.name}) Failed to decode response as JSON.")
            print(f"  ({self.name}) Raw response: {response_str[:100]}...")
            return {"error": "Failed to decode Judge's response.", "raw_response": response_str}

# --- Orchestrator ---

class DetectionOrchestrator:
    """Manages the workflow of the purely zero-shot multi-agent system."""
    def __init__(self, base_url: str = None):
        print("Initializing Multi-Agent Detection System...")
        if base_url: print(f"Using custom base URL: {base_url}")
        try:
            self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY"))
            self.observers = [
                ProbabilityObserver(self.client, AGENT_MODEL),
                RewritabilityObserver(self.client, AGENT_MODEL),
                SyntacticObserver(self.client, AGENT_MODEL)
            ]
            self.judge = JudgeAgent(self.client, AGENT_MODEL)
            print(f"System initialized with Zero-Shot Judge: {self.judge.name}")
        except Exception as e:
            print(f"FATAL: Could not initialize system. Error: {e}")
            self.client = None
            self.observers = []
            self.judge = None

    def run_detection(self, text: str) -> dict:
        """Runs the full detection pipeline on a given text."""
        if not self.client:
            print("System is not initialized. Aborting detection.")
            return {}
        
        print("\n--- Beginning detection process ---")
        print(f"Text to analyze: '{text[:50]}...' ({len(text)} chars)")
        
        print("\n--- Running Observer Analyses ---")
        reports = {}
        for obs in self.observers:
            print(f"\nRunning {obs.name}...")
            reports[obs.name] = obs.analyze(text)
            
        print("\n--- All observer analyses complete. Requesting final judgment ---")
        final_judgment = self.judge.decide(reports)
        
        print("\n--- Detection Results ---")
        if "error" in final_judgment:
            print(f"ERROR: {final_judgment['error']}")
        else:
            try:
                prediction = final_judgment.get("final_prediction", "Unknown")
                confidence = final_judgment.get("final_confidence_score", -1)
                print(f"VERDICT: {prediction} (Confidence: {confidence:.4f})")
                print(f"\nREASONING:\n{final_judgment.get('detailed_reasoning', 'No reasoning provided.')[:300]}...")
            except Exception as e:
                print(f"Error displaying results: {e}")
                print(f"Raw judgment: {final_judgment}")
                
        return final_judgment

# --- Evaluation Workflow ---

def load_and_sample_raid_data(samples_per_class=500, seed=42):
    """Loads RAID from HF, creates a balanced sample of human and gpt-4 texts."""
    print("Loading 'liamdugan/raid' dataset from Hugging Face Hub...")
    try:
        dataset = load_dataset("liamdugan/super-clean-raid", split="train") 
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting to load dataset without specifying a split...")
        try:
            dataset = load_dataset("liamdugan/super-clean-raid")['train']
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            return None
    
    print("Filtering for 'human' samples...")
    human_ds = dataset.filter(lambda x: x['model'] == 'human')
    print("Filtering for 'gpt4' samples...")
    gpt4_ds = dataset.filter(lambda x: x['model'] == 'gpt4')

    human_shuffled = human_ds.shuffle(seed=seed)
    gpt4_shuffled = gpt4_ds.shuffle(seed=seed)

    num_human = min(samples_per_class, len(human_shuffled))
    num_gpt4 = min(samples_per_class, len(gpt4_shuffled))
    print(f"Selecting {num_human} random human samples and {num_gpt4} random gpt-4 samples...")
    
    human_samples = human_shuffled.select(range(num_human))
    gpt4_samples = gpt4_shuffled.select(range(num_gpt4))

    print("Combining and shuffling the final balanced dataset...")
    combined_ds = concatenate_datasets([human_samples, gpt4_samples])
    return combined_ds.shuffle(seed=seed)

def evaluate_system_on_hf_dataset(orchestrator: DetectionOrchestrator, samples_per_class: int, output_file: str):
    """
    Evaluates the given detection orchestrator on a balanced sample from the RAID dataset.
    """
    print("\n--- Starting System Evaluation on Hugging Face Dataset ---")
    
    sampled_dataset = load_and_sample_raid_data(samples_per_class=samples_per_class)
    
    if sampled_dataset is None:
        print("Dataset loading failed. Aborting evaluation.")
        return
    
    print(f"\n--- Running detection on {len(sampled_dataset)} balanced samples ---")
    predictions = []
    true_labels = []

    for item in tqdm(sampled_dataset, desc="Processing samples"):
        try:
            text_to_check = item['generation']
            if not text_to_check or len(text_to_check.split()) < 10:
                continue 

            judgment = orchestrator.run_detection(text_to_check)
            
            if "error" not in judgment:
                if "human_confidence_score" in judgment:
                    # We need to invert the score since the ROC AUC calculation expects higher values for AI
                    predictions.append(1.0 - judgment["human_confidence_score"])
                    true_labels.append(0 if item['model'] == 'human' else 1)
                elif "final_confidence_score" in judgment:
                    predictions.append(judgment["final_confidence_score"])
                    true_labels.append(0 if item['model'] == 'human' else 1)
                else:
                    print(f"Skipping sample due to missing score in judgment")
            else:
                print(f"Skipping sample due to error in judgment: {judgment.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Critical error processing a sample: {e}. Skipping.")
            continue
    
    print("\n--- Evaluating Results ---")
    if len(predictions) > 1 and len(set(true_labels)) > 1:
        roc_auc = roc_auc_score(true_labels, predictions)
        print(f"\nEVALUATION COMPLETE!")
        print(f"ROC AUC Score on balanced Human vs. GPT-4 set: {roc_auc:.4f}")
    else:
        print(f"Could not compute ROC AUC. Processed {len(predictions)} valid samples with labels: {set(true_labels)}.")

    results_df = pd.DataFrame({'true_label': true_labels, 'predicted_ai_confidence': predictions})
    results_df.to_csv(output_file, index=False)
    print(f"Detailed evaluation results saved to {output_file}")


# --- Main Execution ---

if __name__ == "__main__":
    
    # --- WORKFLOW 1: Using the default Zero-Shot System ---
    print("\n=== Running Single Detection with Zero-Shot System ===")
    orchestrator = DetectionOrchestrator(base_url=OPENAI_BASE_URL)
    
    if orchestrator.judge:
        print("\n\n=== TEST 1: HUMAN TEXT ===")
        human_text = ""
        human_result = orchestrator.run_detection(human_text)
        
        print("\n\n=== TEST 2: AI TEXT ===")
        ai_text = "The implementation of advanced methodologies in organizational structures facilitates the optimization of operational efficiency. Strategic deployment of resources across multiple domains enables entities to achieve synergistic outcomes. Furthermore, the utilization of data-driven decision-making processes contributes to the enhancement of performance metrics."
        ai_result = orchestrator.run_detection(ai_text)
        
        print("\n\n=== DETECTION TESTS COMPLETE ===")
    else:
        print("ERROR: Orchestrator could not be initialized. Tests aborted.")
    
    # --- WORKFLOW 2: Running the Hugging Face Evaluation Script ---
    # This section demonstrates how to benchmark the system.
    # It is commented out by default to avoid long runtimes on execution.
    # To run, ensure you have enough API credits and uncomment the lines below.

    # print("\n--- Starting System Benchmarking ---")
    # if orchestrator.judge:
    #     evaluate_system_on_hf_dataset(
    #         orchestrator=orchestrator,
    #         samples_per_class=100, # Use a smaller number for a quick test
    #         output_file="multi_agent_system_raid_eval.csv"
    #     )
    # else:
    #     print("Skipping evaluation because the orchestrator could not be initialized.")

