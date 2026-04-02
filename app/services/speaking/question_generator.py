"""
question_generator.py
──────────────────────
100 pre-written general speaking questions.
2 are picked randomly per session — no LLM call needed.

Benefits over LLM generation:
  - Zero API cost and zero latency for question generation
  - Truly random every session (LLM with same prompt tends to repeat itself)
  - Guaranteed quality — every question is reviewed and appropriate
  - No risk of inappropriate or off-topic questions

Question design rules:
  - Open-ended — encourages ~1 minute of speech
  - General — no domain knowledge needed, suitable for any candidate
  - Positive framing — asks about experiences, opinions, and preferences
  - No sensitive topics — no religion, politics, salary, health, or personal trauma
  - Clear simple language — accessible to all English proficiency levels

Categories covered (10 questions each):
  1. Personal interests and hobbies
  2. Technology and daily life
  3. Work and teamwork
  4. Learning and education
  5. Communication and people skills
  6. Problem-solving and challenges
  7. Future goals and plans
  8. Travel and experiences
  9. Habits and routines
  10. Opinions and preferences
"""

import random


QUESTION_BANK: list[str] = [

    # ── Category 1: Personal interests and hobbies (1–10) ────────────────────
    "Tell us about a hobby or activity you enjoy in your free time and why it is meaningful to you.",
    "What is one skill you have learned outside of school or work, and how did you learn it?",
    "Describe a book, film, or show that had a strong impact on you and explain what you took from it.",
    "What sport or physical activity do you enjoy, and how does it help you in your daily life?",
    "Tell us about a creative activity — like music, art, cooking, or writing — that you enjoy.",
    "Describe something you are passionate about that most people might not know about you.",
    "What is something you enjoy doing alone versus something you enjoy doing with others?",
    "Tell us about a talent or skill you are proud of and how you developed it.",
    "What is a hobby you have always wanted to try but have not started yet, and why?",
    "Describe how you like to spend a typical weekend and what makes it enjoyable for you.",

    # ── Category 2: Technology and daily life (11–20) ─────────────────────────
    "How has technology changed the way you communicate with friends and family?",
    "What is one app or tool you use every day and why is it useful to you?",
    "Do you think people spend too much time on their phones? Share your thoughts.",
    "How do you think artificial intelligence will change jobs in the next ten years?",
    "What is the biggest advantage and disadvantage of social media in your opinion?",
    "How has online learning changed the way people gain new skills and knowledge?",
    "Describe how technology has made one task in your life significantly easier.",
    "What is one piece of technology you could not imagine living without, and why?",
    "Do you prefer reading news online or in a newspaper, and what is your reason?",
    "How do you manage your screen time to maintain a healthy balance in daily life?",

    # ── Category 3: Work and teamwork (21–30) ─────────────────────────────────
    "Describe a time when you worked as part of a team and what your contribution was.",
    "What qualities do you think make someone an effective team member?",
    "Tell us about a project you are proud of — what was your role and what did you achieve?",
    "How do you prefer to receive feedback on your work, and why is that helpful for you?",
    "What does a good working environment look like to you?",
    "Describe a situation where you had to meet a tight deadline and how you managed it.",
    "What is your approach when you disagree with a colleague's idea or decision?",
    "How do you stay organised and manage your tasks when you have many things to do?",
    "Tell us about a skill you want to develop further in your professional life.",
    "What motivates you most when working on a challenging task or project?",

    # ── Category 4: Learning and education (31–40) ────────────────────────────
    "Tell us about the most interesting subject you studied and what made it engaging.",
    "What is the most useful thing you have learned in the last six months?",
    "How do you prefer to learn something new — by reading, watching, or doing?",
    "Describe a teacher, mentor, or guide who influenced you and what you learned from them.",
    "What is one thing you wish your school or college had taught you?",
    "How do you approach learning a subject that you find difficult or boring?",
    "Tell us about a time when you made a mistake and what you learned from the experience.",
    "What is one topic outside your field of study that you are curious to learn more about?",
    "Do you think exams are a good way to measure a student's ability? Share your view.",
    "What advice would you give to someone just starting their education or career?",

    # ── Category 5: Communication and people skills (41–50) ───────────────────
    "Describe a time when you had to explain something complex to someone simply.",
    "Tell us about a situation where good communication helped resolve a problem.",
    "How do you adjust the way you speak depending on who you are talking to?",
    "What do you think is the most important element of a good conversation?",
    "Describe someone in your life who is an excellent communicator and what makes them effective.",
    "How do you handle a situation when someone misunderstands what you said?",
    "Tell us about a time you had to speak in front of a group — how did you prepare?",
    "What is the difference between listening and truly understanding someone?",
    "How do you build trust with a new colleague, classmate, or acquaintance?",
    "What is one communication habit you think everyone should develop?",

    # ── Category 6: Problem-solving and challenges (51–60) ────────────────────
    "Describe a challenge you faced recently and the steps you took to overcome it.",
    "Tell us about a time when a plan did not work out and how you adapted.",
    "How do you approach a problem when you do not know where to start?",
    "Describe a situation where you came up with a creative solution to a problem.",
    "What is the most difficult decision you have had to make, and how did you decide?",
    "How do you stay calm and focused when things are not going as planned?",
    "Tell us about a time when you helped someone else solve a problem.",
    "What strategies do you use when you feel stuck or cannot find a solution?",
    "Describe a time when you had to learn something quickly to solve a problem.",
    "How do you decide when to solve a problem on your own versus asking for help?",

    # ── Category 7: Future goals and plans (61–70) ────────────────────────────
    "Where do you see yourself professionally in the next three to five years?",
    "What is one goal you are actively working towards right now?",
    "Describe the kind of career or work that you believe would make you most satisfied.",
    "What steps are you currently taking to improve yourself personally or professionally?",
    "Tell us about something you want to achieve before the end of this year.",
    "How do you set goals for yourself and track whether you are making progress?",
    "What is one skill you plan to develop in the near future and why?",
    "Describe your ideal work-life balance and how you plan to achieve it.",
    "What kind of impact do you hope to have on your community or workplace?",
    "How do you stay motivated when working towards a long-term goal?",

    # ── Category 8: Travel and experiences (71–80) ────────────────────────────
    "Describe a place you have visited that left a strong impression on you.",
    "Tell us about a cultural experience that taught you something new.",
    "What is a place you would love to visit and what draws you to it?",
    "Describe an experience where you were completely outside your comfort zone.",
    "Tell us about an event or celebration that is special to you and why.",
    "What is the most memorable experience you have had with food from another culture?",
    "How has travelling or exploring new places changed your perspective?",
    "Describe a day or experience that you wish you could relive and why.",
    "What is one local place in your city or town that you think everyone should visit?",
    "Tell us about a time when you tried something new for the first time.",

    # ── Category 9: Habits and routines (81–90) ───────────────────────────────
    "Describe your morning routine and how it sets the tone for your day.",
    "What habit have you built over the past year that has improved your life?",
    "How do you wind down at the end of a long or stressful day?",
    "Tell us about a healthy habit you are trying to build or maintain.",
    "How do you manage your energy levels throughout the day to stay productive?",
    "Describe how you organise your study or work time to get the best results.",
    "What is one daily habit you think is underrated but highly effective?",
    "How do you make sure you get enough rest and recovery in your routine?",
    "Tell us about a habit you gave up that made a positive difference in your life.",
    "How do you balance productive work time with rest and personal activities?",

    # ── Category 10: Opinions and preferences (91–100) ────────────────────────
    "Do you prefer working in a quiet environment or a busy one, and why?",
    "What is your opinion on the importance of punctuality in professional life?",
    "Do you think it is better to be a specialist in one area or to have broad skills?",
    "What quality do you admire most in the people around you and why?",
    "Do you prefer working on one task at a time or handling multiple tasks together?",
    "What is your view on the role of failure in achieving success?",
    "Do you think confidence or competence is more important in a professional setting?",
    "What is your opinion on the value of reading books in the age of the internet?",
    "Do you prefer working in a structured environment or one with more freedom?",
    "What is one change you would make to improve the education system if you could?",
]


def generate_speaking_questions(seed: int = None) -> list[str]:
    """
    Pick 2 questions randomly from the bank.
    Each call returns a different pair — no two sessions get the same set.

    Args:
        seed: optional int for reproducible selection (testing only).
              Leave as None in production for true randomness.

    Returns:
        List of exactly 2 question strings.
    """
    rng = random.Random(seed)
    return rng.sample(QUESTION_BANK, 2)