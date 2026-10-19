# NoteWise

NoteWise turns lecture-note PDFs into study materials. Upload your notes to get:

- A short summary and a list of key concepts
- Related topics grouped from the document
- A fill-in-the-blank quiz
- Answers with relevant passages from your notes

Documents are processed on your computer.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

You can try the sample notes from the sidebar, or upload your own PDF.

## Run tests

```bash
python -m pytest tests/test_nlp.py -v
```
