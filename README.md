# steps to implement

## 1. Clone the repository

```
git clone https://github.com/Abhi202505/proofOfConcept.git
cd proofOfConcept
```

## 2. Create a virtual environment

```
python -m venv .venv
.\.venv\Scripts\activate
```

## 3. Install dependencies
```
pip install -r requirements.txt
```
    Note: If you encounter errors related to transformers or nltk (like hanging downloads or zip errors), you may need to run this command once to pre-load the necessary data:
    python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

## 4. Set up environment variables

Create a file named .env in the root directory and add your API keys:
```
SARVAM_API_KEY=your_sarvam_key_here
GOOGLE_API_KEY=your_gemini_key_here
```

## 6. Run the server
```
python server.py
```

## 7. Access the Client

Open your browser and go to:
```
http://localhost:7860
```