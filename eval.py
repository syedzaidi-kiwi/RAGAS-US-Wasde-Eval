import os
from dotenv import load_dotenv
from datasets import load_dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
import pandas as pd

# Load environment variables from the .env file
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure that the API key is set correctly
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Load the synthetic dataset from Hugging Face
dataset = load_dataset("syedzaidi-kiwi/US_WASDE_Eval", split="test")

# Transform the context column into a list of strings for contexts
def transform_context(context):
    if isinstance(context, str):
        return context.split('|||')
    return context

dataset = dataset.map(lambda x: {'contexts': transform_context(x['context'])})

# Add a placeholder 'answer' column if it doesn't exist
def add_placeholder_answer(example):
    if 'answer' not in example or example['answer'] is None:
        example['answer'] = "Placeholder answer"
    return example

dataset = dataset.map(add_placeholder_answer)

# Ensure the necessary columns are present
required_columns = ['question', 'contexts', 'ground_truth', 'answer']
missing_columns = [col for col in required_columns if col not in dataset.column_names]
if missing_columns:
    raise ValueError(f"Dataset is missing the following required columns: {missing_columns}")

# Perform the evaluation using Ragas
result = evaluate(
    dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

# Print the evaluation results
print(result)

# Convert the results to a pandas DataFrame for further analysis
df_result = result.to_pandas()

# Beautify the DataFrame by renaming columns if necessary and ensuring it looks good
df_result.columns = [col.replace('_', ' ').title() for col in df_result.columns]

# Save the evaluation results to a .csv file
output_results_path = "/Users/kiwitech/Downloads/WASDE/evaluation_results.csv"
df_result.to_csv(output_results_path, index=False)

print(f"Evaluation results have been saved to {output_results_path}")
