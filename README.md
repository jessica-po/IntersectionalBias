============================================
Are Contextual AI Models Biased on Software Engineering Implementation Choices?
============================================

For far, this Github contains this script which generates short, technology-neutral software engineering sentence frames for use in the SE bias dataset. 

Each frame contains a {TECH} placeholder that can be replaced with different technologies from the same category.


The current categories include the seven OSI layers as well as programming languages, databases, cloud development tools, web technologies, and IDEs/editors.


# Setup
Install the required packages:

pip install openai groq pandas python-dotenv

Add the API keys for the models you plan to use to your .env file.

# Running

For example:
python generate_se_frames.py --n 25 --models gpt56 gpt4o llama70b llama8b mistral

You can also run only specific categories:
python generate_se_frames.py --categories X_cloud_development X_language X_database --n 25

Or generate only the OSI categories:
python generate_se_frames.py --osi-only --n 25


The script currently supports GPT-5.6, GPT-4o, Llama 3.3 70B, Llama 3.1 8B, and Mistral Large.