import os

os.environ["OPENAI_API_KEY"] = ""

from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("/Users/kiwitech/Downloads/WASDE")
documents = loader.load()

for document in documents:
    document.metadata['filename'] = document.metadata['source']

    from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

testset.to_pandas()

# Save the DataFrame to a .txt file in the root directory
output_file_path = "/Users/kiwitech/Downloads/WASDE/synthetic_dataset.txt"
df.to_csv(output_file_path, sep='\t', index=False)

print(f"Synthetic dataset has been saved to {output_file_path}")