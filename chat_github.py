import pickle
import os
from llama_index import GPTSimpleVectorIndex

assert (
    os.getenv("OPENAI_API_KEY") is not None
), "Please set the OPENAI_API_KEY environment variable."

from llama_index import download_loader
download_loader("GithubRepositoryReader")

from llama_index.readers.llamahub_modules.github_repo import GithubClient, GithubRepositoryReader

docs = None

docs = None
if os.path.exists("docs.pkl"):
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)

if docs is None:
    print('doc is none')
    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
    loader = GithubRepositoryReader(
        github_client,
        owner =                  "akmhmgc",
        repo =                   "qa_embedding",
        filter_directories =     (["gpt_index", "docs"], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = ([".py"], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                True,
        concurrent_requests =    10,
    )

    docs = loader.load_data(branch="main")

    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

index = GPTSimpleVectorIndex(docs)

output = index.query("How do I use this?")
print(output)
