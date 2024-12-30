import aiohttp
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List
import asyncio


async def pubmed_search(query: str, max_results: int = 35) -> List[Dict]:
    """
    Search PubMed for academic articles using E-utilities.
    """
    try:
        api_key = '4e6ad5ec68a6f95b8526b7440dbdcda2a009'
        base_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "api_key": api_key
        }

        request_delay = 0.34
        ssl_context = aiohttp.TCPConnector(ssl=False)

        async with aiohttp.ClientSession(connector=ssl_context) as session:
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

            async with session.get(search_url, params=base_params) as response:
                if response.status == 429:
                    print("Rate limit exceeded. Waiting before retrying...")
                    await asyncio.sleep(2)
                    return []
                elif response.status != 200:
                    print(f"Search API error: Status {response.status}")
                    return []

                try:
                    search_data = await response.json()
                    ids = search_data.get("esearchresult", {}).get("idlist", [])
                except Exception as e:
                    print(f"Error parsing search results: {str(e)}")
                    return []

            if not ids:
                print("No articles found")
                return []

            # Process results in batches
            batch_size = 50
            results = []

            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                await asyncio.sleep(request_delay)

                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(batch_ids),
                    "retmode": "xml",
                    "api_key": api_key
                }

                fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

                async with session.get(fetch_url, params=fetch_params) as response:
                    if response.status != 200:
                        continue

                    content = await response.text()
                    root = ET.fromstring(content)

                    for article in root.findall(".//PubmedArticle"):
                        try:
                            article_data = {
                                "title": article.find(".//ArticleTitle").text if article.find(
                                    ".//ArticleTitle") is not None else "No title available",
                                "abstract": article.find(".//Abstract/AbstractText").text if article.find(
                                    ".//Abstract/AbstractText") is not None else "No abstract available",
                                "journal": article.find(".//Journal/Title").text if article.find(
                                    ".//Journal/Title") is not None else "No journal available",
                                "date": "Unknown",
                                "doi": article.find(".//ArticleId[@IdType='doi']").text if article.find(
                                    ".//ArticleId[@IdType='doi']") is not None else "No DOI available",
                                "first_author": "No author name available",
                                "pmid": article.find(".//PMID").text if article.find(".//PMID") is not None else ""
                            }

                            # Extract date
                            pub_date = article.find(".//PubDate")
                            if pub_date is not None:
                                year = pub_date.find("Year")
                                month = pub_date.find("Month")
                                year_text = year.text if year is not None else "Unknown"
                                month_text = month.text if month is not None else "01"
                                article_data["date"] = f"{year_text}-{month_text}"

                            # Extract first author
                            authors = article.findall(".//Author")
                            if authors:
                                last_name = authors[0].find("LastName")
                                first_name = authors[0].find("ForeName")
                                if last_name is not None and first_name is not None:
                                    article_data["first_author"] = f"{first_name.text} {last_name.text}"
                                    if len(authors) > 1:
                                        article_data["first_author"] += " et al."

                            results.append(article_data)

                        except Exception as e:
                            print(f"Error processing article: {str(e)}")
                            continue

        print(f"Successfully retrieved {len(results)} articles")
        return results

    except Exception as e:
        print(f"Error in PubMed search: {str(e)}")
        return []
