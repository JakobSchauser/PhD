# Submission types:
# ON_TIME = 1
# LATE = 2
# MISSING = 3
# EXCEPTIONAL = 4

from typing import List, Tuple

class submissionType:
    ON_TIME = 1
    LATE = 2
    MISSING = 3
    EXCEPTIONAL = 4

class Candidate:
    def __init__(self, name: str, grades: List[float], submission_type: submissionType):
        self.name = name
        self.grades = grades
        self.submission_type = submission_type

class ProcessedCandidate:
    def __init__(self, average_score: float, grade_class: int, name_valid: bool, candidate_name: str, submission_score: float):
        self.average_score = average_score
        self.grade_class = grade_class
        self.name_valid = name_valid
        self.candidate_name = candidate_name
        self.submission_score = submission_score


def get_score_from_submission_type(submission_type: submissionType) -> float:
    if submission_type == submissionType.ON_TIME:
        return 1.0
    elif submission_type == submissionType.LATE:
        return 0.8
    elif submission_type == submissionType.MISSING  :
        return 0.0
    elif submission_type == submissionType.EXCEPTIONAL:
        return 1.5
    else:
        return 0.0

def calculate_average_grade(grades: List[float]) -> float:
    """Calculate the average grade from a list of student grades."""
    if not grades:
        return 0.0
    return sum(grades) / len(grades)

    
def check_if_name_is_valid(name: str) -> bool:
    """Check if the candidate's name is valid (non-empty)"""
    return len(name) > 0

def process_candidate(candidate: Candidate) -> ProcessedCandidate:
    """Process a candidate's submission to calculate score, class, and description."""
    average_score = calculate_average_grade(candidate.grades)
    grade_class = int(average_score // 10)  # Classify into 0-9 based on score range
    name_valid = check_if_name_is_valid(candidate.name)
    submission_score = get_score_from_submission_type(candidate.submission_type)

    processed_candidate = ProcessedCandidate(average_score, grade_class, name_valid, candidate.name, submission_score)

    return processed_candidate

def is_top_candidate(pc: ProcessedCandidate) -> bool:
    """Determine and return the top candidate based on criteria."""
    if pc.name_valid and pc.grade_class >= 8 and pc.average_score * pc.submission_score >= 70:
        return True
    return False

