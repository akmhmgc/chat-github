import os
import pickle
import streamlit as st
from llama_index import GPTSimpleVectorIndex, download_loader
from llama_index.readers.llamahub_modules.github_repo import GithubClient, GithubRepositoryReader

st.title("OpenAI and GitHub API App")
st.write("Enter your OpenAI API key and GitHub token to start.")

openai_api_key = st.text_input("OpenAI API Key", value="", type="password")
github_token = st.text_input("GitHub Token", value="", type="password")

if openai_api_key and github_token:
    st.write("API keys have been set.")
    
    # Add input fields for repository information
    owner = st.text_input("Owner (required)", value="")
    repo = st.text_input("Repo (required)", value="")
    filter_directories = st.text_input("Filter Directories (optional, comma-separated)", value="")
    filter_file_extensions = st.text_input("Filter File Extensions (optional, comma-separated)", value="")

    # Process the optional fields
    filter_directories = tuple(filter_directories.split(',')) if filter_directories else None
    filter_file_extensions = tuple(filter_file_extensions.split(',')) if filter_file_extensions else None

    if owner and repo:
        # Load the data
        if not os.path.exists("docs.pkl"):
            download_loader("GithubRepositoryReader")
            github_client = GithubClient(github_token)
            loader = GithubRepositoryReader(
                github_client,
                owner=owner,
                repo=repo,
                filter_directories=(filter_directories, GithubRepositoryReader.FilterType.INCLUDE) if filter_directories else None,
                filter_file_extensions=(filter_file_extensions, GithubRepositoryReader.FilterType.INCLUDE) if filter_file_extensions else None,
                verbose=True,
                concurrent_requests=10,
            )
            docs = loader.load_data(branch="main")
            with open("docs.pkl", "wb") as f:
                pickle.dump(docs, f)
        else:
            with open("docs.pkl", "rb") as f:
                docs = pickle.load(f)

        index = GPTSimpleVectorIndex(docs)

        user_question = st.text_input("Enter your question:", value="")
        if user_question:
            output = index.query(user_question)
            st.write("Response:")
            st.markdown(f"<h3 style='font-size: 18px;'>{output}</h3>", unsafe_allow_html=True)
