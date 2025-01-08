import random
from expected_parrot import Survey, Agent, QuestionLinearScale, SurveyRun, Model

import os
os.environ["OPENAI_API_KEY"] = "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"


# Define demographic groups
age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
wealth_brackets = ["low income", "middle income", "high income"]

# Define questions with positive, neutral, and negative tones, now phrased for 1-5 rankings
questions = [
    # 1
    {
        "positive": "How strongly do you agree that AI can enhance our daily lives? (1 = Strongly Disagree, 5 = Strongly Agree)",
        "neutral": "To what extent do you agree or disagree with the statement: 'AI has a role in our daily lives'? (1 = Strongly Disagree, 5 = Strongly Agree)",
        "negative": "How concerned are you about AI's influence on our daily routines? (1 = Not Concerned at All, 5 = Very Concerned)"
    },
    # 2
    {
        "positive": "How strongly do you agree that AI will create new opportunities for work and innovation? (1 = Strongly Disagree, 5 = Strongly Agree)",
        "neutral": "What is your view on AI’s impact on job opportunities? (1 = Very Negative, 5 = Very Positive)",
        "negative": "How concerned are you that AI might lead to job losses in the future? (1 = Not Concerned at All, 5 = Very Concerned)"
    },
    # Continue with similar questions for the rest...
    {
        "positive": "How likely are you to adopt AI tools to make tasks easier in your life? (1 = Very Unlikely, 5 = Very Likely)",
        "neutral": "Do you see yourself using AI tools in your personal or professional life? (1 = Very Unlikely, 5 = Very Likely)",
        "negative": "How concerned are you about using AI tools in your personal life? (1 = Not Concerned at All, 5 = Very Concerned)"
    },
    {
        "positive": "How much do you agree that AI will improve healthcare? (1 = Strongly Disagree, 5 = Strongly Agree)",
        "neutral": "To what extent do you believe AI should play a role in healthcare? (1 = Not at All, 5 = Very Much)",
        "negative": "How concerned are you about the impact of AI on healthcare? (1 = Not Concerned, 5 = Very Concerned)"
    },
    # Continue defining the rest of the questions in the same way...
]

# Create agents with different demographics as parameters, with 100 simulations each
agents = [
    Agent(name=f"Agent_{age}_{wealth}_{i}", traits={"age_group": age, "wealth_bracket": wealth})
    for age in age_groups for wealth in wealth_brackets for i in range(100)
]

# Create survey questions with demographic-specific prompts and randomized tone for ranking
survey_questions = [
    QuestionLinearScale(
        prompt_template="You are a {age_group} with {wealth_bracket} completing a survey about your feelings regarding the future of AI.\n\n{question}",
        scale_labels=[1, 5],
        parameters={"question": random.choice([q["positive"], q["neutral"], q["negative"]])}
    )
    for q in questions
]

# Define GPT-4 model with high temperature for varied responses
model = Model(model_name="gpt-4", temperature=1.0)

# Create a survey and add questions
survey = Survey(
    title="Demographic Survey on AI Sentiments",
    description="Survey analyzing how different demographics feel about the future of AI."
)

for question in survey_questions:
    survey.add_question(question)

# Run the survey for each agent using GPT-4 with high temperature
survey_run = SurveyRun(survey=survey, agents=agents, model=model)

# Collect responses
responses = survey_run.run()

# Optionally, analyze responses with pandas
import pandas as pd

# Transform responses to a DataFrame for easier analysis
data = {
    "age_group": [],
    "wealth_bracket": [],
    "question": [],
    "response": []
}

for agent, agent_responses in responses.items():
    for question, response in agent_responses.items():
        data["age_group"].append(agent.traits["age_group"])
        data["wealth_bracket"].append(agent.traits["wealth_bracket"])
        data["question"].append(question.parameters["question"])
        data["response"].append(response)

df = pd.DataFrame(data)

# Display the data to start analyzing
print(df.head())
