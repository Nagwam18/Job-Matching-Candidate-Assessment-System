import uvicorn
from fastapi import FastAPI, Form
from threading import Thread
import uuid
import nest_asyncio

from app.utils.file_loader import load_file
from app.services.cv_analysis import analyze_cv, extract_cv_score, extract_skills
from app.services.video_analysis import analyze_video
from app.services.verdict import generate_verdict, format_output

nest_asyncio.apply()

app = FastAPI()
results = {}


@app.post("/evaluate")
async def evaluate(
    job_description: str = Form(...),
    cv_path: str = Form(...),
    video_path: str = Form(...)
):
    session_id = str(uuid.uuid4())

    cv_text = load_file(cv_path)
    cv_analysis = analyze_cv(cv_text, job_description)

    cv_score = extract_cv_score(cv_analysis) or 0.0
    matched, missing = extract_skills(cv_analysis)

    video_result = analyze_video(video_path)
    confidence = video_result.get("confidence", 0.0)
    anxiety = video_result.get("anxiety", 0.0)

    verdict, final_score, behavior, video_score = generate_verdict(
        cv_score, matched, missing, confidence, anxiety
    )

    formatted_output = format_output(
        cv_score, video_score, final_score, verdict, behavior
    )

    results[session_id] = {
        "cv_text": cv_text,
        "cv_analysis": cv_analysis,
        "cv_score": cv_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "video_result": video_result,
        "behavior": behavior,
        "final_score": final_score,
        "llm_verdict": verdict,
        "formatted_output": formatted_output
    }

    return {"session_id": session_id, "output": formatted_output}


@app.get("/results")
async def get_results(session_id: str):
    if session_id not in results:
        return {"error": "invalid session id"}
    return results[session_id]


def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8007)


thread = Thread(target=run_api, daemon=True)
thread.start()
