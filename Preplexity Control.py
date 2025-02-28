import os
import re
import time
import logging
import traceback
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load API key from environment variable
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

# Create OpenAI client
client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")


def get_introduction(title, max_retries=3):
    """
    Get a medical research paper introduction for a given title using the Perplexity API.
    """
    logger.info(f"Generating introduction for: \"{title}\"")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "content": (
                f"""Please generate a comprehensive introduction section for a medical research paper, titled:
{title}

Follow these specific instructions:

Topic Introduction: Begin by introducing the general topic. Provide a clear and concise overview, emphasizing what sets the context for the study.

Literature Background: Highlight key findings from the current literature related to the topic. This should include key findings, major challenges, and significant studies identifying the gap. Clearly articulate the research gap to demonstrate that this paper aims to fill that void. This would involve unresolved questions, conflicting data, or insufficient existing evidence.

Study Objective: Articulate the specific objective of the current study. Make sure this is well-designed relative to the identified gap.

Referencing: Every piece of information, data, or claim made in this introduction must be backed by real, credible, and peer-reviewed articles. Use accurate and relevant references. Please provide all the references, and do not tell me to find or look for a reference myself.

Citation Format: Use a numerical citation format. Each reference should be indicated in the text with a number in parentheses (e.g., [1], [2]).

Citations: Conclude the introduction with a complete and structured list of citations used. Each citation should include all necessary details such as authors, title, journal name, year, volume, issue, page numbers, and DOI.

Augmentation: If there are any specific instructions about the gap that the study addresses, the approach, or the impact of the research, the citations you use should be only recent sources and credible.
                """
            ),
        },
    ]

    for attempt in range(max_retries):
        try:
            logger.info(f"Calling Perplexity API (attempt {attempt + 1}/{max_retries})...")
            start_time = time.time()
            response = client.chat.completions.create(
                model="sonar",
                messages=messages,
            )
            api_time = time.time() - start_time
            logger.info(f"API call successful! (took {api_time:.2f} seconds)")
            logger.info(
                f"Response stats: {response.usage.completion_tokens} completion tokens, "
                f"{response.usage.prompt_tokens} prompt tokens"
            )

            content = response.choices[0].message.content

            logger.info("=" * 80)
            logger.info("FULL PERPLEXITY RESPONSE:")
            logger.info("=" * 80)
            logger.info(content)
            logger.info("=" * 80)

            return content, response.citations
        except Exception as e:
            logger.error(f"Error calling API (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error("All retry attempts failed")
                return None, None


def extract_references(text, citations):
    """
    Extract main text and create a mapping of reference links without modifying the original text.
    """
    full_text = text

    logger.info("Citation Links from API:")
    if citations:
        for i, url in enumerate(citations):
            logger.info(f"{i + 1}. {url}")
    else:
        logger.warning("No citation links returned from API")

    # Split the text to analyze any reference sections, if present.
    parts = re.split(r'(?i)## references|references', text, 1)
    if len(parts) > 1:
        logger.info("REFERENCE SECTION FROM TEXT:")
        logger.info(parts[1].strip())
    else:
        logger.warning("No separate reference section found in text")

    # Map citation numbers to links.
    reference_links = {}
    if citations:
        logger.info(f"Mapped {len(citations)} citation links to reference numbers:")
        for i, url in enumerate(citations):
            reference_links[i + 1] = url
            logger.info(f"Reference {i + 1}: {url}")
    else:
        logger.warning("No citation links to map")

    return full_text, reference_links


def save_interim_results(results, filename=None):
    """Save interim results to prevent data loss."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interim_perplexity_results_{timestamp}.xlsx"

    try:
        # Create a deep copy of results to be saved.
        results_copy = [row.copy() for row in results]
        df_interim = pd.DataFrame(results_copy)

        logger.info(f"Excel columns to be saved: {list(df_interim.columns)}")
        logger.info(f"DataFrame shape: {df_interim.shape}")

        df_interim.to_excel(filename, index=False)
        logger.info(f"Interim results saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving interim results: {e}")


def main():
    logger.info("=" * 80)
    logger.info("PERPLEXITY MEDICAL PAPER GENERATOR")
    logger.info("=" * 80)

    # Configuration: input and output file names.
    input_file = "title_sheet.xlsx"
    output_file = "perplexity_literature_review.xlsx"

    try:
        if not os.path.exists(input_file):
            logger.error(f"Input file '{input_file}' not found.")
            return

        df_titles = pd.read_excel(input_file)
        if "titles" not in df_titles.columns:
            logger.error(f"'titles' column not found in {input_file}.")
            return

        titles = df_titles["titles"].tolist()
        logger.info(f"Loaded {len(titles)} titles from {input_file}")
        logger.info("First 3 titles as sample:")
        for i, title in enumerate(titles[:3]):
            logger.info(f"{i + 1}. {title}")
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        return

    results = []
    max_refs = 0

    for i, title in enumerate(tqdm(titles, desc="Processing titles")):
        logger.info("=" * 80)
        logger.info(f"PROCESSING TITLE {i + 1}/{len(titles)}: \"{title}\"")
        logger.info("=" * 80)

        intro_text, citations = get_introduction(title)

        if intro_text and citations:
            full_text, reference_links = extract_references(intro_text, citations)
            row = {"Title": title, "Main Text": full_text}

            logger.info(f"Adding {len(reference_links)} references to Excel row")
            for j, url in reference_links.items():
                row[f"Reference {j}"] = url

            max_refs = max(max_refs, len(reference_links))
            logger.info(f"Max references across all titles so far: {max_refs}")
            logger.info("WHAT'S BEING SAVED TO EXCEL:")
            logger.info(f"  - Title: \"{title}\"")
            logger.info(f"  - Main Text Length: {len(full_text)} characters")
            logger.info(f"  - Reference Count: {len(reference_links)}")

            results.append(row)
            logger.info(f"Successfully added row for title #{i + 1} to results")

            # Save interim results every 10 titles or at the end.
            if (i + 1) % 10 == 0 or (i + 1) == len(titles):
                save_interim_results(results)

            wait_time = 2
            logger.info(f"Waiting {wait_time} seconds before next API call...")
            time.sleep(wait_time)
        else:
            logger.error(f"Failed to get introduction for: {title}")

    if results:
        logger.info("Preparing final Excel file...")

        # Ensure all rows have the same columns.
        for row in results:
            for i in range(1, max_refs + 1):
                if f"Reference {i}" not in row:
                    row[f"Reference {i}"] = ""

        df_results = pd.DataFrame(results)
        logger.info(f"Final DataFrame shape: {df_results.shape}")
        logger.info(f"Final columns: {list(df_results.columns)}")

        df_results.to_excel(output_file, index=False)
        logger.info(f"Final results saved to {output_file}")

        logger.info("Sample of final data (first row):")
        sample_row = df_results.iloc[0] if not df_results.empty else None
        if sample_row is not None:
            logger.info(f"  Title: {sample_row['Title']}")
            logger.info(f"  Main Text (first 300 chars): {sample_row['Main Text'][:300]}...")
            logger.info(f"  Main Text (last 300 chars): ...{sample_row['Main Text'][-300:]}")
            logger.info(f"  Character count: {len(sample_row['Main Text'])}")
            for col in df_results.columns:
                if col.startswith("Reference ") and sample_row[col]:
                    logger.info(f"  {col}: {sample_row[col]}")
    else:
        logger.error("No results to save.")


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        logger.info("=" * 50)
