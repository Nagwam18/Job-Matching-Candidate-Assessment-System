import re
from app.services.llm_model import tokenizer, model


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
3. Extract the **matched skills** (skills present in both CV and JD, considering understanding, synonyms, and related concepts).
4. Extract the **non-matched skills** (skills required in JD but not present in CV).
5. Calculate **CV Score** = (number of matched skills / total skills in JD) * 100, rounded to 1 decimal.

Return ONLY in this format:

Matched Skills
- skill1
- skill2
...

Non-Matched Skills
- skill1
- skill2

CV Score
XX.X% match
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Matched Skills" in full_output:
        result = full_output[full_output.index("Matched Skills"):].strip()
    else:
        result = full_output.strip()

    return result


def extract_cv_score(result_text):
    match = re.search(r"CV Score\s*([\d]+(?:\.\d+)?)%\s*match", result_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def extract_skills(result_text):
    matched = []
    missing = []

    lines = result_text.splitlines()
    mode = None

    for line in lines:
        line = line.strip()

        if line.lower() == "matched skills":
            mode = "matched"
            continue
        elif line.lower() == "missing skills":
            mode = "missing"
            continue
        elif line.lower().startswith("cv score"):
            mode = None
            continue

        if mode == "matched" and line.startswith("-"):
            matched.append(line[1:].strip())
        elif mode == "missing" and line.startswith("-"):
            missing.append(line[1:].strip())

    return matched, missing
