import re
from app.services.llm_model import tokenizer, model


import re
import torch

def analyze_cv(cv_text, jd_text):
    prompt = f"""
You are an expert recruiter.

CV:
{cv_text}

Job Description:
{jd_text}

Instructions:
1. Identify all skills required in the Job Description.
2. Identify all skills mentioned in the CV.
3. Extract the matched skills (skills present in both CV and JD, considering synonyms and related concepts).
4. Extract the non-matched skills (skills required in JD but not present in CV).
5. Calculate CV Score = (number of matched skills / total skills in JD) * 100, rounded to 1 decimal.

Return ONLY in this exact format:

Matched Skills
- skill

Non-Matched Skills
- skill

CV Score
XX.X% match

Rules:
- Do not repeat any skills.
- Do not use placeholders.
- Do not explain anything.
- Output must end after CV Score.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Matched Skills" in output_text:
        output_text = output_text.split("Matched Skills", 1)[1]
        output_text = "Matched Skills\n" + output_text

    return output_text.strip()


def extract_skills(cv_output: str):
    matched, missing = [], []

    matched_section = re.search(
        r"Matched Skills\s*:?(.*?)(Non-Matched Skills)",
        cv_output,
        re.S | re.I
    )

    if matched_section:
        for line in matched_section.group(1).splitlines():
            skill = line.strip("- ").strip()
            if skill:
                matched.append(skill)

    missing_section = re.search(
        r"Non-Matched Skills\s*:?(.*?)(CV Score)",
        cv_output,
        re.S | re.I
    )

    if missing_section:
        for line in missing_section.group(1).splitlines():
            skill = line.strip("- ").strip()
            if skill:
                missing.append(skill)

    return matched, missing

def extract_cv_score(cv_output: str):
    score_match = re.search(
        r"CV Score\s*([\d\.]+)\s*%",
        cv_output,
        re.I
    )
    return float(score_match.group(1)) if score_match else None
