# Submission types:
# ON_TIME = 1
# LATE = 2
# MISSING = 3
# EXCEPTIONAL = 4

def calculate_average_grade(grades):
    """Calculate the average grade from a list of student grades."""
    if not grades:
        return 0.0
    return sum(grades) / len(grades)

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
    
def check_if_name_is_valid(name):
    """Check if the candidate's name is valid (non-empty)"""
    return len(name) > 0

def process_candidate(candidate_name, grades, submission_type):
    """Process a candidate's submission to calculate score, class, and description."""
    average_score = calculate_average_grade(grades)
    submission_score = get_score_from_submission_type(submission_type)  
    grade_class = int(average_score // 10)  # Classify into 0-9 based on score range
    name_valid = check_if_name_is_valid(candidate_name)

    return average_score, grade_class, submission_score, name_valid, candidate_name


def is_top_candidate(average_score, grade_class, submission_score, name_valid, candidate_name):
    """Determine and return the top candidate based on criteria."""
    if name_valid and grade_class >= 8 and average_score * submission_score >= 70:
        return True
    return False

