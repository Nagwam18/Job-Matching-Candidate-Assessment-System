from app.utils.file_loader import load_file
from app.services.cv_analysis import (
    analyze_cv,
    extract_cv_score,
    extract_skills
)
from app.services.video_analysis import analyze_video
from app.services.verdict import generate_verdict, format_output


def run_pipeline(
    cv_path: str,
    jd_path: str,
    video_path: str | None = None
):
    # Load CV & JD
    cv_text = load_file(cv_path)
    jd_text = load_file(jd_path)

    # CV vs JD analysis (LLM)
    analysis_result = analyze_cv(cv_text, jd_text)

    cv_score = extract_cv_score(analysis_result) or 0.0
    matched, missing = extract_skills(analysis_result)

    # Video analysis (optional)
    if video_path:
        video_result = analyze_video(video_path)
        confidence = video_result.get("confidence", 0.0)
        anxiety = video_result.get("anxiety", 0.0)
    else:
        confidence = 0.0
        anxiety = 0.0

    # Final verdict
    verdict, final_score, behavior, video_score = generate_verdict(
        cv_score=cv_score,
        matched=matched,
        missing=missing,
        confidence=confidence,
        anxiety=anxiety
    )

    # Format output
    formatted_output = format_output(
        cv_score=cv_score,
        video_score=video_score,
        final_score=final_score,
        verdict=verdict,
        behavior=behavior
    )

    return {
        "cv_score": cv_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "confidence": confidence,
        "anxiety": anxiety,
        "final_score": final_score,
        "behavior": behavior,
        "verdict": verdict,
        "formatted_output": formatted_output
    }


if __name__ == "__main__":
    result = run_pipeline(
        cv_path="app/data/sample_cv.pdf",
        jd_path="app/data/sample_jd.txt",
        video_path="app/data/sample_video.mp4"
    )

    print(result["formatted_output"])
