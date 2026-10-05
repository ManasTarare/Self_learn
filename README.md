# Pxplore

Pxplore recommends a Microsoft Learn module, then asks an LLM to generate an original, structured course grounded in that module and personalized with your saved learning history. The app uses Streamlit.

## Setup (Windows PowerShell)

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    Copy-Item .env.example .env
    # Add one provider key to .env

Provider priority for automatic selection is Gemini, then Anthropic, then OpenAI. The provider and model can also be selected in the app.

## Prepare course data

    python convert_mslearn.py data/mslearn_course_structures.json data/snippets.jsonl

The repository includes prepared data/snippets.jsonl and data/snippets_full.jsonl. The full file includes lesson text where available. The optional lesson fetch workflow is described in fetch_content.py.

## Run

    streamlit run app.py

On first launch, create a local Me profile with a name, username, password, long-term goal, short-term goal, and starting level. The app stores the account and learning data in data/student_memory.db. Passwords are salted and hashed; they are never stored as plain text. The app supports one local learner account. Use Me → Profile → Erase your profile and all learning data to remove it and its learning history.

## What the app remembers

- Goals, current level, generated courses and each course's level.
- Quiz answers and scores, assignment submissions and LLM feedback.
- Concept mastery estimates and prerequisite links derived from course content and assessment evidence.
- Recent study notes and a tentative preferred teaching approach inferred from repeated learning results.

The knowledge graph is an evolving estimate. Mastery values are updated as new assessment evidence arrives; they are not permanent labels. The inferred teaching approach is a hypothesis, with evidence and confidence shown in the profile. The current level is user-editable; each generated course separately stores its inferred difficulty level.

The SQLite database stays on the machine running Streamlit. For personalization, the app sends relevant goals, study notes, course/assessment summaries, the latest quiz answers/results for preference inference, and (for assignment grading) the submitted assignment to the LLM provider selected in the sidebar. The username, password, and display name are not sent to the model. The SQLite file itself is not encrypted at rest, so keep the project folder on a machine and account you trust.

## Main flow

1. The app retrieves likely Microsoft Learn modules using BM25 keyword ranking and semantic embeddings.
2. An LLM chooses a next module using your profile and saved learning record.
3. A second LLM call creates a course with objectives, lessons, activities, quiz, and assignment.
4. Quiz results and assignment evaluations are saved and update concept mastery estimates.
5. The Me profile shows your learning record and a small course/concept knowledge graph.

Generated courses are original learning material grounded in the catalog module; they aren't official Microsoft Learn courses. Keep attribution for any Microsoft Learn source text you publish. Microsoft Learn content is licensed CC BY 4.0.

## Project files

| File | Purpose |
|---|---|
| app.py | Streamlit login, Learn page, Me profile, course display and assessments |
| learner_store.py | SQLite schema, account hashing, profile, course history and learner graph |
| pipeline.py | Retrieval planning, course generation, assignment evaluation and preference inference |
| retriever.py | BM25 + sentence-transformer hybrid retrieval |
| llm.py | Gemini / Anthropic / OpenAI calls, retries and disk cache |
| convert_mslearn.py | Microsoft Learn catalog JSON to JSONL snippets |
| fetch_content.py | Adds lesson text from a Microsoft Learn repository clone |
| prompts/ | Recommendation, course-generation and assessment prompts |



