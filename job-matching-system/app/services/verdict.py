from app.services.llm_model import tokenizer, model



def generate_verdict(cv_score, matched, missing, confidence, anxiety):
    cv_score = cv_score or 0.0
    confidence = confidence or 0.0
    anxiety = anxiety or 0.0
    matched = matched or []
    missing = missing or []

    behavior = (
        "confident" if confidence > anxiety else
        "anxious" if anxiety > confidence else
        "neutral"
    )

    video_score = confidence
    final_score = round((cv_score * 0.7 + video_score * 0.3), 2)

    if cv_score == 0.0 or not matched:
        verdict_text = (
            f"The candidate has no matching skills in the CV and appears {behavior}. "
            "The candidate is not suitable for the position."
        )
        return verdict_text, final_score, behavior, video_score

    matched_skills = ", ".join(matched)
    missing_skills = ", ".join(missing) if missing else "None"

    prompt = f"""
You are an expert recruiter.

Candidate Evaluation:
- CV Score: {cv_score}%
- Video Behavior: {behavior}
- Matched Skills: {matched_skills}
- Missing Skills: {missing_skills}
- Final Score: {final_score}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.9,
        do_sample=True
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    verdict_text = full_output.strip()

    return verdict_text, final_score, behavior, video_score


def format_output(cv_score, video_score, final_score, verdict, behavior):
    text = f"""Final Score:
• CV Score (70%): {cv_score}
• Video Score (30%): {video_score}
• Candidate behavior: {behavior}
• Final = {final_score}
{verdict}
"""
    return text
