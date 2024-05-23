import os
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from the .env file
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure that the API key is set correctly
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Load documents from directory
loader = DirectoryLoader("/Users/kiwitech/Downloads/WASDE")
documents = loader.load()

# Ensure each document has a filename in its metadata
for document in documents:
    document.metadata['filename'] = document.metadata.get('source', 'unknown')

# Create a test set generator with OpenAI models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", api_key=openai_api_key)
critic_llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key)
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# Generate synthetic test set with 10 samples
testset = generator.generate_with_langchain_docs(
    documents, 
    test_size=50, 
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
)

# Export the test set to a Pandas DataFrame
df = testset.to_pandas()

# Ensure the DataFrame has the required columns
# Check and rename columns if necessary
required_columns = ['question', 'context', 'ground_truth']
df = df.rename(columns={
    'contexts': 'context'  # Adjust according to your actual column names
})

# Convert the DataFrame to a Hugging Face dataset
dataset = Dataset.from_pandas(df[required_columns])

# Create a dataset dictionary
dataset_dict = DatasetDict({"test": dataset})

# Push the dataset to Hugging Face
dataset_dict.push_to_hub("US_WASDE_Eval")

print("Dataset successfully uploaded to Hugging Face.")
