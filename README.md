
# Smart Domain Analyzer

AI-powered tool to extract Domain-Driven Design (DDD) models from natural language business requirements.




## Overview
Smart Domain Analyzer is an AI-powered tool that converts plain-text business requirements into structured Domain-Driven Design (DDD) models. It uses LLMs like LLaMA (via Groq) to extract Domains, Subdomains, Aggregates, Entities, and their Attributes, outputting them in standardized JSON format.

💡 Use Case

Quickly generate draft DDD models from requirement documents to accelerate system design and align with business goals.

🧰 Technologies

- Python: Core programming language for scripting and automation
- LangChain: Framework for orchestrating prompt engineering and LLM chains
- LLaMA: Back-end AI engines for natural language understanding and DDD structure generation



## Features

- Converts business requirements into structured DDD output (Domain, Subdomain, Aggregates, Entities).
- Supports LLaMA3 inference APIs.
- JSON and graphical format output.
- Feedback loop ready for learning from user corrections (future scope).


## Getting Started
Prerequisites
- Python 3.9+
- Local installation of Ollama
## Installation

Install my-project with npm

```bash
git clone https://github.com/bg-ai-hub/smart-domain-analyzer.git
cd smart-domain-analyzer
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
    
## Usage/Examples

```python
python ddd_extrator.py
```


## Contributing

Contributions are always welcome!
- Fork this repo
- Create a feature branch
- Raise a PR with detailed description


