import openai
import requests
import spacy

openai.api_key = ""
diffbot_api_key = ""
# Enter own keys


def get_chatgpt_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=250,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()


def generate_question(previous_questions):
    nlp = spacy.load("en_core_web_md")
    similarity_threshold = 0.8

    prompt = (
        "Generate a unique test question on electrical engineering subjects such as thyristorss. "
        "Then, provide the answer and the source of the answer in a URL format. Format your response "
        "like this:\n\nQuestion: ...\nAnswer: ...\nSource: ...\n\nExample:")

    question_and_answer = get_chatgpt_response(prompt)

    # Extract just the question from the full string
    question = question_and_answer.split('\n')[0]
    question_doc = nlp(question)

    # Check if the generated question is similar to any previous questions
    similar_question_found = any(
        question_doc.similarity(nlp(prev_question)) > similarity_threshold for prev_question in previous_questions)
    while similar_question_found:
        question_and_answer = get_chatgpt_response(prompt)
        question = question_and_answer.split('\n')[0]
        question_doc = nlp(question)
        similar_question_found = any(
            question_doc.similarity(nlp(prev_question)) > similarity_threshold for prev_question in previous_questions)

    # Extract source from the full string
    source_url = question_and_answer.split('\n')[2].replace("Source: ", "").strip()

    return question_and_answer, source_url


def get_text_from_url_diffbot(url: str, api_key: str) -> str:
    api_url = f"https://api.diffbot.com/v3/article?token={api_key}&url={url}"
    response = requests.get(api_url)
    data = response.json()

    if 'objects' in data and len(data['objects']) > 0:
        return data['objects'][0]['text']
    else:
        return ""


def is_valid_url(url):
    try:
        response = requests.get(url, timeout=3)  # Send a request to the URL
        return response.status_code < 400  # Return True if the status code is less than 400
    except (requests.exceptions.RequestException, ValueError):
        return False  # If there was an error with the request, return False

def is_information_from_sources(answer: str, source_urls: list, api_key: str, threshold: float = 0.5) -> bool:
    nlp = spacy.load('en_core_web_md')
    answer_doc = nlp(answer)
    valid_source = False

    # Iterate over all provided source URLs
    for source_url in source_urls:
        if is_valid_url(source_url):  # Only proceed if the URL is valid
            source_text = get_text_from_url_diffbot(source_url, api_key)
            source_doc = nlp(source_text)

            if answer_doc.has_vector and source_doc.has_vector:
                # If the answer is short (less than 5 words), lower the threshold
                similarity_threshold = threshold if len(answer.split()) >= 5 else 0.2
                similarity = answer_doc.similarity(source_doc)
                if similarity >= similarity_threshold:
                    valid_source = True
                    break  # No need to check the remaining sources if we've found one

    return valid_source

if __name__ == "__main__":
    num_questions = 5
    previous_questions = []

    for i in range(num_questions):
        question_and_answer, source_url = generate_question(previous_questions)

        lines = question_and_answer.strip().split("\n")
        question = lines[0].replace("Question: ", "").strip()
        previous_questions.append(question)
        answer = lines[1].replace("Answer: ", "").strip()
        source_urls = lines[2].replace("Source: ", "").strip().split('; ')  # Split multiple URLs by '; '

        result = is_information_from_sources(answer, source_urls, diffbot_api_key)
        print(f"Q{i + 1}: {question}")
        print(f"A{i + 1}: {answer}")
        if not is_valid_url(source_url):
            print("Note: The provided source URL is invalid. The answer was generated based on the model's training data.")
        print(f"Source: {source_urls}")
        print("Is information from source?", "Yes" if result else "No")
        print("\n")
