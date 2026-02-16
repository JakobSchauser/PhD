# Submission types:
# ON_TIME = 1
# LATE = 2
# MISSING = 3
# EXCEPTIONAL = 4

from typing import List, Tuple

def calculate_average_grade(grades: List[float]) -> float:
    """Calculate the average grade from a list of student grades."""
    if not grades:
        return 0.0
    return sum(grades) / len(grades)

def check_if_name_is_valid(name: str) -> bool:
    """Check if the candidate's name is valid (non-empty)"""
    return len(name) > 0

def get_score_from_submission_type(submission_type: int) -> float:
    if submission_type == 1:
        return 1.0
    elif submission_type == 2:
        return 0.8
    elif submission_type == 3:
        return 0.0
    elif submission_type == 4:
        return 1.5
    else:
        return 0.0


def process_candidate(candidate_name: str, grades: List[float], submission_type: int) -> Tuple[float, int, bool, str]:
    """Process a candidate's submission to calculate score, class."""
    average_score = calculate_average_grade(grades)
    grade_class = int(average_score // 10)  # Classify into 0-9 based on score range
    name_valid = check_if_name_is_valid(candidate_name)

    return average_score, grade_class, name_valid, candidate_name

def is_top_candidate_anonymized(average_score: float, grade_class: int, name_valid: bool, candidate_name: str) -> bool:
    """Determine and return the top candidate based on criteria."""
    if name_valid and grade_class >= 8 and average_score >= 80:
        return True
    return False


