import re



def load_text(path):
    """Read the data text from a given path"""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            corpus_text = file.read()
        return corpus_text
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return ""

def parse_corpus(corpus_text):
    """Parse out the corpus by storing each section/video in a dictionary with the video ID being its key and 
    the transcript being the value"""

    # Define a pattern
    pattern = r"Title: (.*?)\nVideo ID: (.*?)\nTranscript:\n(.*?)(?=Title:|$)"
    matches = re.findall(pattern, corpus_text, re.DOTALL)

    # Distribute the video and its transcript into a list of dictionaries
    video_and_transcript_dict = []
    for section in matches:
        # For each section, store the id with the second element and transcript with the 3rd element of each section
        video_and_transcript_dict.append({
            "id": section[1].strip(),
            "transcript": section[2].strip()
        })    

    return video_and_transcript_dict

def filter_special_characters(section, min_words):
    """For any given section, filter out the '------' and get rid of section that has less than """
    """Returns a filtered section"""
    # Replace '-----' with ''
    filtered_section = section['transcript'].replace('--------------------------------------------------------------------------------', '').strip()

    # Replace '[Music]' with ''
    filtered_section = filtered_section.replace('[Music]', '').strip()

    # Count up the length of transcript value
    transcript_len = len(filtered_section.strip())
    if transcript_len > min_words:
        return filtered_section
    return None
        
def filter_text(sections, min_words=20):
    """Go through each section and filters out the '------' and get rid of sections that have less than 10 words in the transcript key"""
    filtered_sections = []
    for section in sections:
        filtered_section = filter_special_characters(section, min_words)
        # If the section meets the criteria
        if filtered_section:
            section['transcript'] = filtered_section
            filtered_sections.append(section)
    
    return filtered_sections


PATH = '/Users/kenlam/Desktop/Data science/ML projects/RAG_resume/model/corpus_from_web_crawler.txt'
corpus = load_text(PATH)
sections = parse_corpus(corpus)
filtered_text = filter_text(sections)
print(filtered_text[70]['transcript'])
