import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

def call_llm(prompt, max_retries=3):
    """Call Groq API and return raw text response, with retries on failure."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.9
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

def extract_json(text):
    """Extract JSON object from LLM response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  JSON parse failed: {e}")
        return None
# ── JD Generation ──────────────────────────────────────────

JD_ROLE_DISTRIBUTION = {
    "Data Analyst": 5,
    "Data Scientist": 5,
    "ML Engineer": 4,
    "Backend Developer": 5,
    "Software Engineer": 5,
    "Frontend Developer": 4,
    "Full Stack Developer": 4,
    "DevOps Engineer": 3,
    "Cloud Engineer": 3,
    "Business Analyst": 2,
    "QA / Test Automation Engineer": 3,
    "Mobile App Developer": 4,
    "Cybersecurity Analyst": 3,
}

JD_SCHEMA_EXAMPLE = """{
  "jd_id": "PLACEHOLDER",
  "role_title": "%s",
  "company_type": "<Startup / MNC / Product-based / Service-based>",
  "required_skills": ["<5-8 specific must-have technical skills/tools>"],
  "preferred_skills": ["<2-4 nice-to-have skills>"],
  "min_years_experience": <integer 0-3>,
  "min_education": "<degree requirement>",
  "responsibilities": ["<3-5 realistic bullet points>"]
}"""

def build_jd_prompt(role_title):
    return f"""Generate a realistic job description for a {role_title} position at a randomly chosen company_type (Startup / MNC / Product-based / Service-based).

Return ONLY valid JSON matching this exact schema, no markdown blocks, no extra text:
{JD_SCHEMA_EXAMPLE % role_title}

Constraints:
- required_skills must be specific, named tools/technologies (e.g. "Python", "PostgreSQL", "Docker") — never vague terms like "programming" or "databases"
- responsibilities should reflect real day-to-day work for this exact role
- Vary min_years_experience and company_type naturally across generations
"""

def generate_all_jds():
    all_jds = []
    jd_counter = 1
    for role, count in JD_ROLE_DISTRIBUTION.items():
        print(f"Generating {count} JDs for: {role}")
        for i in range(count):
            prompt = build_jd_prompt(role)
            raw = call_llm(prompt)
            if raw is None:
                print(f"  Skipped (API failure) for {role} #{i+1}")
                continue
            parsed = extract_json(raw)
            if parsed is None:
                print(f"  Skipped (parse failure) for {role} #{i+1}")
                continue
            parsed["jd_id"] = f"J{jd_counter:04d}"
            all_jds.append(parsed)
            jd_counter += 1
            time.sleep(1)  # be polite to rate limits
    return all_jds

if __name__ == "__main__":
    # Quick test with one simple call
    test_prompt = 'Return ONLY this JSON, nothing else: {"status": "ok", "test": true}'
    raw = call_llm(test_prompt)
    print("Raw response:", raw)
    parsed = extract_json(raw)
    print("Parsed:", parsed)
if __name__ == "__main__":
    print("Generating JDs...")
    jds = generate_all_jds()
    print(f"\nSuccessfully generated {len(jds)} / 50 JDs")
    
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/jds.json", "w", encoding="utf-8") as f:
        json.dump(jds, f, indent=2)
    print("Saved to data/raw/jds.json")