# seo_analyzer_app.py
# Streamlit application for Single Page SEO Analysis (No PageSpeed Insights)
# Version: 2025-04-28 (Gauge Fix + Modernization)

import streamlit as st
import requests
from requests import Session # Explicit import for clarity
from requests.exceptions import RequestException, Timeout, TooManyRedirects, ConnectionError
from bs4 import BeautifulSoup
import pandas as pd
import altair as alt
from urllib.parse import urlparse, urljoin
import json
import time
import os
from readability import Document  # Uses readability-lxml
import re
import validators  # For URL validation
import traceback  # For detailed error logging
from typing import List, Dict, Tuple, Optional, Any, Union

# --- Configuration & Constants ---
USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
REQUEST_TIMEOUT: int = 15  # seconds

# SEO Best Practice Thresholds
TITLE_MAX_LEN: int = 60
META_DESC_MAX_LEN: int = 160
MIN_FONT_SIZE_PX: int = 16  # General recommendation - Basic Check Only Now

# API Keys from Environment Variables (SERP API only now)
# For Streamlit Cloud deployment, set these as Secrets in the app settings
SERP_API_KEY: Optional[str] = os.environ.get('SERP_API_KEY') or st.secrets.get("SERP_API_KEY")
SERP_API_ENDPOINT: Optional[str] = os.environ.get('SERP_API_ENDPOINT') or st.secrets.get("SERP_API_ENDPOINT")

# Type Alias for Result Dictionaries
ResultDict = Dict[str, Dict[str, Any]]

# --- Helper Functions ---

def is_valid_url(url: str) -> bool:
    """Validate the URL format."""
    try:
        return bool(validators.url(url))
    except Exception:
        return False

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_html(url: str) -> Tuple[Optional[str], Optional[int], Optional[float], Optional[str]]:
    """
    Fetches HTML content of a URL.

    Args:
        url: The URL to fetch.

    Returns:
        A tuple containing:
        - HTML content (str) or None if failed.
        - HTTP status code (int) or None if failed before response.
        - Time To First Byte (float, seconds) or None if failed.
        - Error message (str) or None if successful.
    """
    headers: Dict[str, str] = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        # Try to detect encoding, fall back to apparent_encoding or utf-8
        response.encoding = response.apparent_encoding if response.encoding == 'ISO-8859-1' else response.encoding or 'utf-8'

        # Basic TTFB timing (Not a replacement for CWV!)
        ttfb: float = response.elapsed.total_seconds()
        return response.text, response.status_code, ttfb, None  # content, status_code, ttfb, error
    except Timeout:
        return None, None, None, f"Request timed out after {REQUEST_TIMEOUT} seconds."
    except RequestException as e:
        status_code: Optional[int] = getattr(e.response, 'status_code', None)
        error_msg: str = f"Could not fetch URL: {e}"
        st.error(f"Network error fetching {url}: {e}") # Show network errors more prominently
        return None, status_code, None, error_msg
    except Exception as e:
        return None, None, None, f"An unexpected error occurred during fetch: {e}"

def get_domain(url: str) -> Optional[str]:
    """Extracts the domain name from a URL."""
    try:
        return urlparse(url).netloc
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_robots_txt(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetches and returns robots.txt content for the domain of the URL.

    Args:
        url: The URL whose domain's robots.txt should be fetched.

    Returns:
        A tuple containing:
        - robots.txt content (str) or None if error. Empty string if 4xx (treated as allowed).
        - Error message (str) or None if successful or 4xx.
    """
    domain: Optional[str] = get_domain(url)
    if not domain:
        return None, "Invalid URL provided."

    # Construct robots.txt URL carefully (handle potential ports, ensure https/http)
    parsed_url = urlparse(url)
    scheme: str = parsed_url.scheme or 'https'
    robots_url: str = f"{scheme}://{domain}/robots.txt"

    headers: Dict[str, str] = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(robots_url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.text, None # Success
        elif 400 <= response.status_code < 500:
             # Treat 4xx errors (like 404 Not Found, 403 Forbidden) as permissive (allow crawling)
            return "", f"Robots.txt not found or inaccessible (Status: {response.status_code}). Assuming allowed."
        else:
            # Handle other errors (e.g., 5xx server errors)
            return None, f"Could not fetch robots.txt (Status: {response.status_code})."
    except RequestException as e:
        return None, f"Could not fetch robots.txt: {e}"
    except Exception as e:
       return None, f"Unexpected error fetching robots.txt: {e}"

def is_disallowed(robots_content: Optional[str], target_url_path: str, user_agent: str = USER_AGENT) -> bool:
    """
    Checks if a specific path is disallowed by robots.txt content.
    Focuses on rules for '*' and the specified user_agent.
    Handles basic wildcards (*) and end-of-string ($). VERY basic implementation.

    Args:
        robots_content: The content of the robots.txt file.
        target_url_path: The path part of the URL to check (e.g., '/page').
        user_agent: The user agent string to check rules for.

    Returns:
        True if the path is disallowed for the user agent, False otherwise.
    """
    if robots_content is None or not target_url_path:
        return False  # Assume allowed if no robots.txt or path

    # Ensure path starts with /
    if not target_url_path.startswith('/'):
        target_url_path = '/' + target_url_path

    lines: List[str] = robots_content.splitlines()
    relevant_rules: Dict[str, Dict[str, List[str]]] = {'*': {'allow': [], 'disallow': []}}
    specific_rules: Dict[str, Dict[str, List[str]]] = {} # Store specific agent rules
    current_agents: List[str] = []
    ua_lower: str = user_agent.lower()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        try:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == 'user-agent':
                agent: str = value.lower()
                current_agents = [] # Start a new agent block
                if agent == '*':
                    current_agents.append('*')
                # Only track rules for our specific agent if defined
                if agent == ua_lower:
                    if agent not in specific_rules:
                        specific_rules[agent] = {'allow': [], 'disallow': []}
                    current_agents.append(agent)

            elif key in ['allow', 'disallow'] and value and current_agents:
                rule_path: str = value
                # Store rules for currently active user agents ('*' or specific)
                for agent_key in current_agents:
                    if agent_key == '*':
                        if rule_path not in relevant_rules['*'][key]:
                            relevant_rules['*'][key].append(rule_path)
                    elif agent_key in specific_rules: # Should always be ua_lower if present
                        if rule_path not in specific_rules[agent_key][key]:
                            specific_rules[agent_key][key].append(rule_path)

        except ValueError:
            continue # Ignore lines that don't split correctly

    # Determine rules applying to our specific user agent (specific rules override '*')
    # Combine rules, specific agent's rules come second (implicitly higher priority by matching first below)
    final_rules: Dict[str, List[str]] = {
        'allow': relevant_rules['*']['allow'] + specific_rules.get(ua_lower, {}).get('allow', []),
        'disallow': relevant_rules['*']['disallow'] + specific_rules.get(ua_lower, {}).get('disallow', [])
    }

    # Sort rules by length descending (most specific first)
    # Google/Bing prioritize based on length of the path prefix
    final_rules['allow'].sort(key=len, reverse=True)
    final_rules['disallow'].sort(key=len, reverse=True)

    longest_allow_match_len: int = -1
    longest_disallow_match_len: int = -1

    # Check Disallow rules (most specific matching rule wins)
    for rule in final_rules['disallow']:
        pattern_str: str = re.escape(rule).replace(r'\*', '.*')
        if rule.endswith('$') and pattern_str.endswith(r'\$'):
             pattern_str = pattern_str[:-2] + '$' # Match end of string
        # No need for complex path matching logic if using regex match from start
        try:
            if re.match(pattern_str, target_url_path):
                longest_disallow_match_len = len(rule)
                break # First (most specific) match found
        except re.error:
            continue # Ignore invalid regex patterns

    # Check Allow rules (most specific matching rule wins)
    for rule in final_rules['allow']:
        pattern_str = re.escape(rule).replace(r'\*', '.*')
        if rule.endswith('$') and pattern_str.endswith(r'\$'):
             pattern_str = pattern_str[:-2] + '$'
        try:
            if re.match(pattern_str, target_url_path):
                longest_allow_match_len = len(rule)
                break # First (most specific) match found
        except re.error:
            continue

    # Googlebot logic: Disallowed if a disallow rule matches AND (no allow rule matches OR the matching allow rule is less specific)
    if longest_disallow_match_len != -1: # A disallow rule matches
        if longest_allow_match_len == -1: # No allow rule matches
            return True
        elif longest_allow_match_len < longest_disallow_match_len: # Matching allow rule is less specific
            return True

    return False # Allowed by default or by a more specific allow rule

@st.cache_data(ttl=86400) # Cache for a day
def check_broken_links(links_to_check: List[str]) -> List[Tuple[str, str]]:
    """
    Checks a list of absolute HTTP/HTTPS links for non-200 status codes using HEAD/GET requests.

    Args:
        links_to_check: A list of absolute URLs (strings).

    Returns:
        A list of tuples, where each tuple contains (broken_link_url, status_or_error_string).
    """
    broken: List[Tuple[str, str]] = []
    # Ensure links are unique and absolute before checking
    unique_links: List[str] = sorted(list(set(
        link for link in links_to_check
        if isinstance(link, str) and link.startswith(('http://', 'https://'))
    )))

    if not unique_links:
        return []

    headers: Dict[str, str] = {'User-Agent': USER_AGENT}
    # Use a session for potential connection reuse and better handling
    with Session() as session:
        session.headers.update(headers)
        session.max_redirects = 5 # Limit redirects to avoid loops

        # Consider using st.progress for visual feedback if list is long
        # progress_bar = st.progress(0)
        # total_links = len(unique_links)

        for i, link in enumerate(unique_links):
            status: Optional[str] = None
            error_type: Optional[str] = None
            try:
                # Try HEAD request first (faster)
                response = session.head(link, timeout=10, allow_redirects=True)
                if not response.ok: # Check status code >= 400
                    # Fallback to GET if HEAD fails or is disallowed (e.g., 405 Method Not Allowed)
                    # Check for common HEAD failures or client errors
                    if response.status_code in [405, 403, 501] or str(response.status_code).startswith('4'):
                        try:
                            # stream=True avoids downloading large files, just check status
                            response_get = session.get(link, timeout=12, stream=True, allow_redirects=True)
                            if not response_get.ok:
                                status = f"GET Status: {response_get.status_code}"
                            response_get.close() # Ensure connection is closed
                        except RequestException as e_get:
                            error_type = f"GET Error: {type(e_get).__name__}"
                    else: # Keep the original HEAD status if it wasn't a likely HEAD issue (e.g., 5xx)
                        status = f"HEAD Status: {response.status_code}"

            except Timeout:
                error_type = "Timeout"
            except TooManyRedirects:
                error_type = "Too Many Redirects"
            except ConnectionError:
                error_type = "Connection Error"
            except RequestException as e:
                 # Store status code from exception response if available
                err_status = getattr(e.response, 'status_code', None)
                status_info = f" (Status: {err_status})" if err_status else ""
                error_type = f"Error: {type(e).__name__}{status_info}"
            except Exception as e: # Catch any other unexpected errors
                error_type = f"Unexpected Error: {type(e).__name__}"

            if status or error_type:
                broken.append((link, status or error_type or "Unknown Error")) # Ensure error_type has value

            # Update progress bar if implemented
            # progress_bar.progress((i + 1) / total_links)
            time.sleep(0.05) # Slightly reduce delay, be mindful of target servers

    return broken


# --- API Call Functions (Placeholder for Competitor Analysis) ---
@st.cache_data(ttl=3600)
def get_serp_competitors_placeholder(
    keywords: List[str],
    target_domain: str,
    country: str = 'us',
    lang: str = 'en'
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """
    Placeholder function for fetching competitor data via a SERP API.
    Replace this with actual SERP API calls if you have credentials.
    Requires SERP_API_KEY and SERP_API_ENDPOINT secrets/env vars.

    Args:
        keywords: List of target keywords.
        target_domain: The domain of the target URL to exclude from results.
        country: Search country code (e.g., 'us', 'gb').
        lang: Search language code (e.g., 'en', 'es').

    Returns:
        A tuple containing:
        - List of competitor dictionaries (keys: position, title, link, domain).
        - Dictionary of extracted SERP features.
        - Error message (str) or None if successful (or using placeholder).
    """
    # Display warning only if API keys are not set AND this function is called
    if not SERP_API_KEY or not SERP_API_ENDPOINT:
        st.warning(f"""
        **Competitor Identification Disabled:**
        This requires a paid SERP API (e.g., SerpApi, ValueSERP, ScraperAPI) integration.
        Using placeholder data for demonstration. Set `SERP_API_KEY` and `SERP_API_ENDPOINT`
        secrets in Streamlit Cloud settings and update this function to enable real competitor fetching.
        """)
        # Return empty/placeholder data if keys are missing
        competitors_list_sim = []
        serp_features_sim = {}
        return competitors_list_sim, serp_features_sim, "SERP API credentials not configured."


    # --- If API keys ARE set, proceed with placeholder or REAL call ---
    # Simulate finding competitors for the first keyword if real call is commented out
    primary_keyword: str = keywords[0] if keywords else "seo"
    st.write(f"(Attempting competitor search for: '{primary_keyword}' using configured API)") # Indicate API use

    # --- Example using a hypothetical SERP API call ---
    # # UNCOMMENT AND ADAPT THIS BLOCK FOR YOUR SPECIFIC SERP API PROVIDER
    # params = {
    #     'api_key': SERP_API_KEY,
    #     'q': primary_keyword,
    #     'location': country, # Adjust param names based on API provider (e.g., 'google_domain', 'gl', 'country')
    #     'hl': lang,          # Adjust param name (e.g., 'hl', 'language')
    #     'num': 10            # Get top 10 results
    # }
    # try:
    #     response = requests.get(SERP_API_ENDPOINT, params=params, timeout=20)
    #     response.raise_for_status()
    #     serp_data = response.json() # Or handle based on API response format
    #
    #     # --- PARSE REAL API RESPONSE ---
    #     # This part depends heavily on the specific SERP API's output format
    #     # Example structure (ADAPT TO YOUR API):
    #     organic_results = serp_data.get('organic_results', [])
    #     competitors = []
    #     for r in organic_results:
    #         link = r.get('link')
    #         domain = get_domain(link)
    #         if link and domain and domain != target_domain: # Exclude target domain
    #             competitors.append({
    #                 'position': r.get('position'),
    #                 'title': r.get('title'),
    #                 'link': link,
    #                 'domain': domain
    #             })
    #             if len(competitors) >= 5: # Take top 5 competitors
    #                 break
    #
    #     # Extract other SERP features based on your API's output
    #     serp_features = {
    #         'search_information': serp_data.get('search_information', {}),
    #         'related_questions': serp_data.get('related_questions', []),
    #         'featured_snippet': serp_data.get('featured_snippet', None),
    #         'knowledge_graph': serp_data.get('knowledge_graph', None),
    #         'inline_videos': serp_data.get('inline_videos', None),
    #         'top_stories': serp_data.get('top_stories', None),
    #         'local_pack': serp_data.get('local_pack', None),
    #         # Add other features your API provides
    #     }
    #     return competitors, serp_features, None # Success
    #
    # except RequestException as e:
    #      st.error(f"SERP API request failed: {e}")
    #      return [], {}, f"SERP API request failed: {e}"
    # except Exception as e:
    #      st.error(f"Error processing SERP API response: {e}")
    #      return [], {}, f"Error processing SERP API response: {e}"
    # --- End of Hypothetical API Call ---


    # --- Fallback to Placeholder data IF API call above is commented out OR fails ---
    # (This part will be reached if the API block is commented or if it errors out without returning)
    st.warning("Using **placeholder** competitor data as real API call is commented out or failed.")
    competitors_list_sim = [
        {'position': 1, 'title': f'Ultimate Guide to {primary_keyword} - Competitor A', 'link': 'https://competitor-a.com/ultimate-guide', 'domain': 'competitor-a.com'},
        {'position': 2, 'title': f'{primary_keyword} Strategies for 2025 - Competitor B', 'link': 'https://competitor-b.com/strategies', 'domain': 'competitor-b.com'},
        {'position': 3, 'title': f'Why {primary_keyword} Matters - Competitor C Blog', 'link': 'https://blog.competitor-c.com/why-seo', 'domain': 'blog.competitor-c.com'},
    ]
    serp_features_sim = {
        'search_information': {'query_displayed': primary_keyword},
        'organic_results': [c for c in competitors_list_sim if c['domain'] != target_domain][:3], # Simulate structure
        'related_questions': [
            {'question': f'What is {primary_keyword}?'}, {'question': f'How to improve {primary_keyword}?'}, {'question': f'Best tools for {primary_keyword}?'}
        ] if 'seo' in primary_keyword else [],
        'featured_snippet': None, 'knowledge_graph': None, 'inline_videos': None, 'top_stories': None, 'local_pack': None,
    }
    # Filter out target domain from the simulation list itself
    actual_competitors_sim = [c for c in competitors_list_sim if c.get('domain') != target_domain]
    return actual_competitors_sim[:3], serp_features_sim, None # Return top 3 simulated competitors


# --- Analysis Functions ---

def analyze_on_page(
    soup: BeautifulSoup,
    url: str,
    keywords: List[str],
    html_content: str
) -> Tuple[ResultDict, List[str], List[str]]:
    """
    Analyzes On-Page SEO elements of the parsed HTML.

    Args:
        soup: BeautifulSoup object of the page HTML.
        url: The URL being analyzed.
        keywords: List of target keywords.
        html_content: Raw HTML content (used for readability).

    Returns:
        A tuple containing:
        - Dictionary of on-page analysis results.
        - List of on-page recommendations.
        - List of all found absolute HTTP/HTTPS links on the page.
    """
    results: ResultDict = {}
    recommendations: List[str] = []
    base_domain: Optional[str] = get_domain(url)

    # 1. Title Tag
    try:
        title_tag = soup.find('title')
        title_text: str = title_tag.string.strip() if title_tag and title_tag.string else ""
        title_len: int = len(title_text)
        results['title_presence'] = {'value': bool(title_text), 'pass': bool(title_text), 'details': title_text}
        results['title_length'] = {'value': title_len, 'pass': 0 < title_len <= TITLE_MAX_LEN, 'details': f"Length: {title_len} chars (Recommended: 10-{TITLE_MAX_LEN})"}
        title_kw_found: List[str] = [kw for kw in keywords if kw.lower() in title_text.lower()] if title_text else []
        results['title_keywords'] = {'value': bool(title_kw_found), 'pass': bool(title_kw_found), 'details': f"Keywords Found: {', '.join(title_kw_found) if title_kw_found else 'None'}"}
        modifiers: List[str] = ['guide', 'best', 'review', 'how to', 'checklist', 'tutorial', str(pd.Timestamp.now().year)]
        title_modifier_found: bool = any(mod in title_text.lower() for mod in modifiers) if title_text else False
        results['title_modifiers'] = {'value': title_modifier_found, 'pass': title_modifier_found, 'details': "Check for engaging words (e.g., 'guide', 'best', year)."}

        if not bool(title_text): recommendations.append("CRITICAL: Add a compelling Title Tag.")
        elif not results['title_length']['pass']: recommendations.append(f"Improve Title Tag: Adjust length ({title_len} chars) to be between 10 and {TITLE_MAX_LEN} characters.")
        if bool(title_text) and not results['title_keywords']['pass']: recommendations.append("Improve Title Tag: Include primary target keywords near the beginning.")
        if bool(title_text) and not results['title_modifiers']['pass']: recommendations.append("Consider Title Modifiers: Add terms like 'guide', 'best', year, etc. to increase CTR.")

    except Exception as e:
        st.error(f"Error analyzing Title tag: {e}")
        results['title_presence'] = results['title_length'] = results['title_keywords'] = results['title_modifiers'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    # 2. Meta Description
    try:
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        meta_desc_content: str = meta_desc_tag.get('content', '').strip() if meta_desc_tag else ""
        meta_desc_len: int = len(meta_desc_content)
        results['meta_desc_presence'] = {'value': bool(meta_desc_content), 'pass': bool(meta_desc_content), 'details': meta_desc_content[:100] + "..." if meta_desc_content else "Missing"}
        results['meta_desc_length'] = {'value': meta_desc_len, 'pass': 50 < meta_desc_len <= META_DESC_MAX_LEN, 'details': f"Length: {meta_desc_len} chars (Recommended: ~50-{META_DESC_MAX_LEN})"}
        meta_kw_found: List[str] = [kw for kw in keywords if kw.lower() in meta_desc_content.lower()] if meta_desc_content else []
        results['meta_desc_keywords'] = {'value': bool(meta_kw_found), 'pass': bool(meta_kw_found), 'details': f"Keywords Found: {', '.join(meta_kw_found) if meta_kw_found else 'None'}"}

        if not bool(meta_desc_content): recommendations.append("CRITICAL: Add a unique and informative Meta Description.")
        elif not results['meta_desc_length']['pass']: recommendations.append(f"Improve Meta Description: Adjust length ({meta_desc_len} chars) to be ~50-{META_DESC_MAX_LEN} characters.")
        if bool(meta_desc_content) and not results['meta_desc_keywords']['pass']: recommendations.append("Improve Meta Description: Include target keywords naturally.")
    except Exception as e:
        st.error(f"Error analyzing Meta Description: {e}")
        results['meta_desc_presence'] = results['meta_desc_length'] = results['meta_desc_keywords'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    # 3. Header Tags (H1-H6)
    try:
        headers: Dict[str, List[str]] = {f'H{i}': [] for i in range(1, 7)}
        for i in range(1, 7):
            for header in soup.find_all(f'h{i}'):
                headers[f'H{i}'].append(header.get_text(strip=True))

        h1_tags: List[str] = headers['H1']
        num_h1: int = len(h1_tags)
        results['h1_presence'] = {'value': num_h1, 'pass': num_h1 == 1, 'details': f"Found {num_h1} H1 tag(s). Recommended: Exactly 1."}
        h1_text: str = h1_tags[0] if num_h1 >= 1 else "" # Take first H1 if multiple exist for keyword check
        h1_kw_found: List[str] = [kw for kw in keywords if kw.lower() in h1_text.lower()] if h1_text else []
        results['h1_keywords'] = {'value': bool(h1_kw_found), 'pass': bool(h1_kw_found), 'details': f"Keywords in (first) H1: {', '.join(h1_kw_found) if h1_kw_found else 'None'}"}
        results['h1_content'] = {'value': h1_text, 'pass': bool(h1_text), 'details': h1_text[:100] + "..." if h1_text else "N/A"}
        has_subheadings: bool = any(len(headers[f'H{i}']) > 0 for i in range(2, 7))
        results['h2_h6_structure'] = {'value': has_subheadings, 'pass': has_subheadings, 'details': f"Found H2s: {len(headers['H2'])}, H3s: {len(headers['H3'])}, etc. Used for structure?"}

        if num_h1 == 0: recommendations.append("CRITICAL: Add an H1 tag that accurately describes the page content.")
        elif num_h1 > 1: recommendations.append("CRITICAL: Ensure there is exactly one H1 tag on the page.")
        if num_h1 == 1 and not h1_kw_found: recommendations.append("Improve H1 Tag: Include primary target keywords.")
        if not has_subheadings: recommendations.append("Improve Content Structure: Use H2-H6 tags to organize content logically.")
    except Exception as e:
        st.error(f"Error analyzing Header tags: {e}")
        results['h1_presence'] = results['h1_keywords'] = results['h1_content'] = results['h2_h6_structure'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    # 4. Content Body & Readability
    main_content_text: str = ""
    word_count: int = 0
    readability_error: Optional[str] = None
    try:
        # Use readability-lxml for better main content extraction
        doc = Document(html_content)
        summary_html: str = doc.summary()
        summary_soup = BeautifulSoup(summary_html, 'lxml') # Parse the extracted HTML
        main_content_text = summary_soup.get_text(separator=' ', strip=True)
        word_count = len(main_content_text.split())
    except Exception as e:
        readability_error = f"Readability processing failed: {e}. Analysis might be less accurate."
        st.warning(readability_error + " Falling back to full body text.")
        # Fallback to trying full body text if readability fails
        try:
            # Extract text from the original soup, potentially less accurate
            main_content_text = soup.get_text(separator=' ', strip=True)
            word_count = len(main_content_text.split())
        except Exception as e_fallback:
            error_msg = f"Could not extract text content: {e_fallback}"
            st.error(error_msg)
            main_content_text = ""
            word_count = 0
            # Ensure keys exist even on complete failure
            results['content_word_count'] = {'value': 0, 'pass': False, 'details': error_msg}
            results['keyword_density'] = {'value': 'N/A', 'pass': False, 'details': 'N/A'}
            results['keyword_prominence'] = {'value': 'N/A', 'pass': False, 'details': 'N/A'}
            results['readability'] = {'value': 'N/A', 'pass': False, 'details': 'N/A'}

    # Analyze content only if text was extracted
    if word_count > 0:
        results['content_word_count'] = {'value': word_count, 'pass': word_count >= 300, 'details': f"{word_count} words (approx. main content). Minimum ~300 recommended. Compare vs competitors."}
        keyword_counts: Dict[str, int] = {kw: main_content_text.lower().count(kw.lower()) for kw in keywords}
        total_kw_count: int = sum(keyword_counts.values())
        density: float = (total_kw_count / word_count * 100) if word_count > 0 else 0.0
        # Keyword Density: Generally avoid specific targets, focus on natural use. Pass/Fail removed.
        results['keyword_density'] = {'value': f"{density:.2f}%", 'pass': True, 'details': f"Keywords: {keyword_counts}. Density approx {density:.2f}%. Focus on natural use, avoid stuffing."}
        first_100_words: str = " ".join(main_content_text.split()[:100])
        kw_in_first_100: List[str] = [kw for kw in keywords if kw.lower() in first_100_words.lower()]
        results['keyword_prominence'] = {'value': bool(kw_in_first_100), 'pass': bool(kw_in_first_100), 'details': f"Keywords in first 100 words: {', '.join(kw_in_first_100) if kw_in_first_100 else 'None'}."}

        if word_count < 500: recommendations.append("Improve Content Depth: Content seems thin (<500 words). Consider expanding if appropriate for the topic.")
        if not results['keyword_prominence']['pass']: recommendations.append("Improve Keyword Placement: Try to include a primary keyword naturally within the first paragraph/100 words.")

        # Simplified Readability Metrics (using basic counts)
        try:
            sentences: int = len(re.findall(r'[.!?]+', main_content_text)) # Approx sentence count
            sentences = max(1, sentences) # Avoid division by zero
            words: int = word_count
            avg_sentence_length: float = words / sentences
            # Paragraphs based on double newlines (might not be accurate depending on source HTML structure)
            paragraphs: int = main_content_text.count('\n\n') + 1
            paragraphs = max(1, paragraphs) # Avoid division by zero
            avg_paragraph_length: float = words / paragraphs

            # Basic pass/fail based on common recommendations
            readability_pass = avg_sentence_length < 25 and avg_paragraph_length < 150
            results['readability'] = {
                'value': f"~{avg_sentence_length:.1f} words/sentence, ~{avg_paragraph_length:.0f} words/paragraph",
                'pass': readability_pass,
                'details': "Lower is often better. Check for short sentences/paragraphs, lists, formatting. (Consider integrating 'textstat' library for Flesch score)."
            }
            if not readability_pass: recommendations.append("Improve Readability: Break down long sentences/paragraphs. Use lists, bolding, and whitespace.")
            if readability_error: # Append the earlier warning here
                 results['readability']['details'] += f" ({readability_error})"
        except Exception as e:
            readability_calc_error = f"Could not calculate basic readability metrics: {e}"
            results['readability'] = {'value': 'Error', 'pass': False, 'details': readability_calc_error}
            recommendations.append(f"Warning: {readability_calc_error}")

    elif 'content_word_count' not in results: # Handle case where no text could be extracted at all
        error_msg = "Could not extract page content for analysis."
        results['content_word_count'] = {'value': 0, 'pass': False, 'details': error_msg}
        results['keyword_density'] = {'value': 'N/A', 'pass': False, 'details': 'N/A'}
        results['keyword_prominence'] = {'value': 'N/A', 'pass': False, 'details': 'N/A'}
        results['readability'] = {'value': 'N/A', 'pass': False, 'details': 'N/A'}
        recommendations.append(f"CRITICAL: {error_msg}")


    # 5. Image SEO
    try:
        images: List[Any] = soup.find_all('img')
        num_images: int = len(images)
        results['image_count'] = {'value': num_images, 'pass': True, 'details': f"Found {num_images} image tag(s)."}

        if num_images > 0:
            alt_missing: int = 0
            alt_generic: int = 0
            descriptive_filenames: int = 0
            # Expanded list of potentially generic/non-descriptive terms
            generic_alts_lower: List[str] = ['image', 'picture', 'logo', 'icon', 'graphic', 'photo', 'alt', 'spacer', 'banner', 'button', 'img', '']
            generic_filenames_parts: List[str] = ['image', 'pic', 'logo', 'icon', 'banner', 'photo', 'img', 'bg', 'background', 'thumb', 'sprite', 'spacer', 'placeholder', 'default', 'temp', 'unnamed', 'untitled', 'graphic']

            for img in images:
                alt_text: str = img.get('alt', '').strip()
                alt_lower: str = alt_text.lower()

                if not alt_text: # Alt text is completely missing
                    alt_missing += 1
                elif len(alt_text) < 5 or alt_lower in generic_alts_lower: # Alt text is short or generic
                    alt_generic += 1
                # Note: A purely decorative image SHOULD have alt=""

                src: Optional[str] = img.get('src')
                if src:
                    try:
                         # Get filename part, remove query params, remove extension, convert to lower
                        filename_base: str = src.split('/')[-1].split('?')[0].split('.')[0].lower()
                        # Check if filename contains non-generic parts and is reasonably long
                        if filename_base and len(filename_base) > 3 and not any(gf_part in filename_base for gf_part in generic_filenames_parts):
                             # Simple check: avoids filenames like 'img_123' or 'logo-new' but allows 'black-cat-sleeping'
                             # Could be improved with more sophisticated NLP/checks
                             descriptive_filenames += 1
                    except Exception:
                        pass # Ignore errors parsing specific filenames

            # Calculate percentages
            alt_text_present_count = num_images - alt_missing
            alt_text_coverage: float = (alt_text_present_count / num_images * 100) if num_images > 0 else 100.0
            descriptive_filename_coverage: float = (descriptive_filenames / num_images * 100) if num_images > 0 else 100.0

            # Alt text check: Aim for high coverage, penalize missing more than generic
            results['image_alt_text'] = {
                'value': f"{alt_text_coverage:.0f}% Present",
                'pass': alt_text_coverage >= 90, # High threshold for presence
                'details': f"{alt_text_present_count}/{num_images} images have alt text. Missing: {alt_missing}, Potentially Generic/Short: {alt_generic}. (Note: Decorative images should have alt=\"\")"
            }
            # Filename check: Less critical but good practice
            results['image_filenames'] = {
                'value': f"{descriptive_filename_coverage:.0f}% Descriptive",
                'pass': descriptive_filename_coverage >= 75, # Moderate threshold
                'details': f"~{descriptive_filenames}/{num_images} images seem to have descriptive filenames (e.g., 'black-cat.jpg' vs 'img_123.jpg')."
            }

            if alt_text_coverage < 90: recommendations.append("Improve Image SEO: Add descriptive alt text to all meaningful images. Use alt=\"\" for decorative ones.")
            if alt_text_coverage >= 90 and alt_generic > 0 : recommendations.append("Refine Image Alt Text: Review the potentially generic/short alt texts for better descriptions.")
            if descriptive_filename_coverage < 75: recommendations.append("Improve Image SEO: Use descriptive file names for images (e.g., 'keyword-rich-name.jpg').")

        else: # No images found
            results['image_alt_text'] = {'value': 'N/A', 'pass': True, 'details': "No images found."}
            results['image_filenames'] = {'value': 'N/A', 'pass': True, 'details': "No images found."}

        # General image recommendations
        recommendations.append("Optimize Images: Ensure images are compressed and use modern formats (e.g., WebP, AVIF) for faster loading (Manual Check Required).")
        recommendations.append("Optimize Images: Check image dimensions are appropriate for their display size (Manual Check Required).")

    except Exception as e:
        st.error(f"Error analyzing Images: {e}")
        results['image_count'] = results['image_alt_text'] = results['image_filenames'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    # 6. Links
    internal_links: List[Dict[str, str]] = []
    external_links: List[Dict[str, str]] = []
    all_links_href: List[str] = [] # Store absolute URLs for broken link check
    try:
        anchor_tags = soup.find_all('a', href=True)
        for link in anchor_tags:
            href: Optional[str] = link.get('href')
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue # Skip invalid, anchor, JS, mailto, tel links

            try:
                abs_url: str = urljoin(url, href) # Resolve relative URLs
                parsed_abs = urlparse(abs_url)

                # Ensure it's a valid, absolute HTTP/HTTPS URL with a domain
                if parsed_abs.scheme in ['http', 'https'] and parsed_abs.netloc:
                    all_links_href.append(abs_url) # Add to list for checking
                    link_domain: str = parsed_abs.netloc
                    anchor_text: str = link.get_text(strip=True)

                    # Classify as internal or external
                    if link_domain == base_domain:
                        internal_links.append({'url': abs_url, 'anchor': anchor_text})
                    elif link_domain and link_domain != base_domain:
                        external_links.append({'url': abs_url, 'anchor': anchor_text})
            except Exception:
                 # st.warning(f"Could not parse or resolve link: {href}") # Optional: log parsing issues
                 continue # Ignore errors resolving/parsing specific URLs

        num_internal: int = len(internal_links)
        num_external: int = len(external_links)

        # Internal link check depends on content length
        internal_link_pass: bool = num_internal > 0 if word_count >= 300 else True
        results['internal_links_count'] = {'value': num_internal, 'pass': internal_link_pass, 'details': f"Found {num_internal} internal links."}
        results['external_links_count'] = {'value': num_external, 'pass': True, 'details': f"Found {num_external} external links (to other domains)."}

        # Sample Anchors (show unique, non-empty anchors)
        internal_anchors_sample: List[str] = sorted(list(set(
            (l['anchor'][:50] + ('...' if len(l['anchor']) > 50 else '')).replace('\n', ' ').strip()
            for l in internal_links if l.get('anchor')
        )))[:5] # Limit sample size
        external_anchors_sample: List[str] = sorted(list(set(
             (l['anchor'][:50] + ('...' if len(l['anchor']) > 50 else '')).replace('\n', ' ').strip()
             for l in external_links if l.get('anchor')
        )))[:5]

        results['internal_anchors'] = {'value': ', '.join(internal_anchors_sample) if internal_anchors_sample else "None Found", 'pass': True, 'details': "Sample internal link anchors. Check for relevance and diversity, avoid 'click here'."}
        results['external_anchors'] = {'value': ', '.join(external_anchors_sample) if external_anchors_sample else "None Found", 'pass': True, 'details': "Sample external link anchors. Ensure they link to relevant, authoritative sources if used."}

        if not internal_link_pass: recommendations.append("Improve Internal Linking: Add relevant internal links to other pages on your site.")
        # Only recommend external links if content is substantial and none exist - subjective
        # if num_external == 0 and word_count > 1000: recommendations.append("Consider External Linking: If appropriate, link out to relevant, authoritative external resources.")

        # Broken Links (checked separately later, provide placeholder)
        results['broken_links'] = {'value': 'Pending check...', 'pass': True, 'details': 'Checking links in background...' }

    except Exception as e:
        st.error(f"Error analyzing Links: {e}")
        results['internal_links_count'] = results['external_links_count'] = results['internal_anchors'] = results['external_anchors'] = results['broken_links'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    return results, recommendations, all_links_href


def analyze_technical(
    soup: BeautifulSoup,
    url: str,
    html_content: str, # Raw content for basic checks
    ttfb: Optional[float], # Pass TTFB if available
    keywords: List[str]
) -> Tuple[ResultDict, List[str]]:
    """
    Analyzes Technical SEO aspects (excluding PageSpeed Insights).

    Args:
        soup: BeautifulSoup object of the page HTML.
        url: The URL being analyzed.
        html_content: Raw HTML string for basic text checks.
        ttfb: Time To First Byte in seconds, if measured.
        keywords: List of target keywords (for URL check).

    Returns:
        A tuple containing:
        - Dictionary of technical analysis results.
        - List of technical recommendations.
    """
    results: ResultDict = {}
    recommendations: List[str] = []
    parsed_url = urlparse(url)

    # 1. Page Load Speed (Basic TTFB Only)
    try:
        ttfb_pass: Optional[bool] = (ttfb < 0.8) if ttfb is not None else None
        ttfb_value: str = f"{ttfb:.3f}s" if ttfb is not None else "N/A"
        results['ttfb'] = {'value': ttfb_value, 'pass': ttfb_pass, 'details': f"Time To First Byte (TTFB): {ttfb_value}. (Target <0.8s. This is NOT Core Web Vitals or full load speed)."}
        if ttfb is not None and not ttfb_pass:
            recommendations.append("Improve Server Response Time (TTFB): TTFB is high (>0.8s). Investigate server performance, caching, database queries, or CDN usage.")
        # Always recommend manual check for full speed insights
        recommendations.append("Review Page Speed Manually: Use tools like Google PageSpeed Insights (web.dev/measure) or WebPageTest.org for detailed speed metrics and Core Web Vitals.")
    except Exception as e:
        st.error(f"Error processing TTFB: {e}")
        results['ttfb'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    # 2. Mobile-Friendliness (Basic Checks - requires manual verification)
    try:
        viewport_tag = soup.find('meta', attrs={'name': 'viewport'})
        viewport_content: Optional[str] = viewport_tag.get('content') if viewport_tag else None
        has_viewport: bool = bool(viewport_tag and viewport_content)
        results['mobile_viewport'] = {'value': 'Present' if has_viewport else 'Missing', 'pass': has_viewport, 'details': f"Viewport meta tag found: {'Yes' if has_viewport else 'No'}. Content: '{viewport_content}'" if has_viewport else "Missing viewport meta tag."}
        if not has_viewport:
            recommendations.append("CRITICAL: Add a viewport meta tag for mobile responsiveness (e.g., <meta name='viewport' content='width=device-width, initial-scale=1'>).")
        elif has_viewport and viewport_content and "user-scalable=no" in viewport_content:
             recommendations.append("Improve Mobile Experience: Avoid 'user-scalable=no' in viewport tag to allow zooming for accessibility.")


        # Basic heuristic checks - require manual validation
        results['mobile_fonts'] = {'value': 'Manual Check', 'pass': True, 'details': f"Manually verify using Browser DevTools that base font size is legible (>= {MIN_FONT_SIZE_PX}px) on mobile devices."}

        # Very basic check for potential interstitials by looking for common keywords
        # This is highly unreliable and needs manual verification.
        interstitial_keywords = ['popup', 'modal', 'interstitial', 'subscribe-box', 'lightbox', 'overlay']
        has_potential_interstitial: bool = any(kw in html_content.lower() for kw in interstitial_keywords)
        results['mobile_interstitials'] = {
            'value': 'Basic Check',
            'pass': True, # Cannot reliably fail this automatically
            'details': f"Code contains keywords possibly related to popups/modals? {'Yes' if has_potential_interstitial else 'No'}. Manually check for intrusive interstitials on mobile."
            }
        if has_potential_interstitial: recommendations.append("Review Mobile Experience: Ensure any popups/interstitials are not intrusive and easily dismissible on mobile devices (Check Google's guidelines).")

        results['mobile_tap_targets'] = {'value': 'Manual Check', 'pass': True, 'details': f"Manually verify using Browser DevTools that buttons/links are large enough and spaced adequately for easy tapping on mobile (e.g., >= 48x48px)."}

    except Exception as e:
        st.error(f"Error analyzing Mobile Friendliness basics: {e}")
        results['mobile_viewport'] = results['mobile_fonts'] = results['mobile_interstitials'] = results['mobile_tap_targets'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    # 3. URL Structure
    try:
        url_str: str = str(url)
        url_len: int = len(url_str)
        results['url_length'] = {'value': url_len, 'pass': url_len < 100, 'details': f"URL Length: {url_len} chars (Shorter, descriptive URLs are generally better)."}
        # Use keywords passed to the function
        url_path_lower: str = parsed_url.path.lower()
        url_kw_found: List[str] = [kw for kw in keywords if kw.lower() in url_path_lower] # Check path mainly
        results['url_keywords'] = {'value': bool(url_kw_found), 'pass': bool(url_kw_found), 'details': f"Keywords in URL path: {', '.join(url_kw_found) if url_kw_found else 'None'}."}

        # Readability check: Hyphens preferred, avoid underscores, parameters, excessive length/depth
        path_part: str = parsed_url.path
        uses_underscores: bool = '_' in path_part
        has_params: bool = bool(parsed_url.query)
        uses_hyphens: bool = '-' in path_part
        readable_structure: bool = uses_hyphens and not uses_underscores and not has_params

        readability_details = []
        if uses_hyphens: readability_details.append("Uses hyphens (good)")
        else: readability_details.append("Consider hyphens")
        if uses_underscores: readability_details.append("Uses underscores (avoid)")
        if has_params: readability_details.append("Contains parameters (avoid if possible)")
        if len(path_part.split('/')) > 5 : readability_details.append("Depth > 5 levels (consider flatter structure)")

        results['url_readability'] = {'value': 'Readable' if readable_structure else 'Less Readable', 'pass': readable_structure, 'details': f"Checks: {'; '.join(readability_details)}."}

        if url_len >= 100: recommendations.append("Consider URL Structure: URL is long (>100 chars). Shorter, focused URLs are preferred if possible.")
        if not results['url_keywords']['pass']: recommendations.append("Improve URL Structure: Include relevant keywords in the URL path (using hyphens).")
        if not readable_structure: recommendations.append("Improve URL Structure: Use hyphens to separate words, avoid underscores or parameters where possible, keep structure relatively flat.")
    except Exception as e:
        st.error(f"Error analyzing URL Structure: {e}")
        results['url_length'] = results['url_keywords'] = results['url_readability'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    # 4. HTTPS Security
    try:
        is_https: bool = parsed_url.scheme == 'https'
        results['https'] = {'value': 'Yes' if is_https else 'No - CRITICAL', 'pass': is_https, 'details': f"URL uses HTTPS: {'Yes' if is_https else 'No. HTTPS is essential for security and SEO.'}"}
        if not is_https: recommendations.append("CRITICAL SECURITY/SEO RISK: Migrate site to HTTPS immediately. Ensure all resources load over HTTPS.")
    except Exception as e:
        st.error(f"Error checking HTTPS: {e}")
        results['https'] = {'value': 'Error', 'pass': False, 'details': str(e)}


    # 5. Schema Markup / Structured Data
    try:
        schema_found: List[str] = []
        # Check for JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                # Handle potential comments within script tags if needed more robustly
                script_content: Optional[str] = script.string
                if script_content:
                    # Clean potential leading/trailing whitespace/comments if necessary
                    clean_content = script_content.strip()
                    # Basic check if it looks like JSON
                    if clean_content.startswith(("{", "[")) and clean_content.endswith(("}", "]")):
                        data: Any = json.loads(clean_content)
                        if isinstance(data, list): # Handle array of schema objects
                            for item in data:
                                schema_type: Optional[str] = item.get('@type') if isinstance(item, dict) else None
                                if schema_type and isinstance(schema_type, str): schema_found.append(f"JSON-LD: {schema_type}")
                        elif isinstance(data, dict): # Handle single schema object
                            schema_type = data.get('@type')
                            if schema_type and isinstance(schema_type, str): schema_found.append(f"JSON-LD: {schema_type}")
            except json.JSONDecodeError:
                 schema_found.append("JSON-LD: Error parsing")
                 st.warning(f"Found JSON-LD script, but failed to parse: {script_content[:100]}...")
            except Exception: # Catch other potential errors during processing
                schema_found.append("JSON-LD: Error reading")

        # Check for Microdata (basic check for itemscope/itemtype)
        microdata_items = soup.find_all(itemscope=True)
        for item in microdata_items:
            item_type_url: Optional[str] = item.get('itemtype')
            if item_type_url and isinstance(item_type_url, str):
                try:
                    # Extract type name after last '/'
                    schema_type_name: str = item_type_url.split('/')[-1]
                    if schema_type_name:
                       schema_found.append(f"Microdata: {schema_type_name}")
                except Exception:
                    pass # Ignore errors parsing itemtype

        # Check for RDFa Lite (basic check for vocab/typeof) - Less common now
        rdfa_items = soup.find_all(property=True, typeof=True) # Look for elements with both
        for item in rdfa_items:
            rdfa_type: Optional[str] = item.get('typeof')
            if rdfa_type and isinstance(rdfa_type, str):
                 schema_found.append(f"RDFa: {rdfa_type}")

        unique_schema_found = sorted(list(set(schema_found)))
        results['schema_markup'] = {'value': ', '.join(unique_schema_found) if unique_schema_found else 'None Found', 'pass': bool(unique_schema_found), 'details': "Presence of Schema.org markup (JSON-LD, Microdata, RDFa). Helps search engines understand content."}
        if not unique_schema_found: recommendations.append("Implement Schema Markup: Add relevant structured data (e.g., Article, BreadcrumbList, FAQPage, Product) using JSON-LD format. Validate with Google's Rich Results Test.")
    except Exception as e:
        st.error(f"Error analyzing Schema Markup: {e}")
        results['schema_markup'] = {'value': 'Error', 'pass': False, 'details': str(e)}


    # 6. Crawlability/Indexability
    try:
        # Robots.txt Check (placeholder to be filled after fetch in main flow)
        results['robots_txt_check'] = {'value': 'Pending check...', 'pass': True, 'details': 'Checking robots.txt allow/disallow rules...'}

        # Meta Robots Tag
        meta_robots = soup.find('meta', attrs={'name': 'robots'})
        robots_content: str = meta_robots['content'].lower().replace(' ', '') if meta_robots and meta_robots.has_attr('content') else ''
        noindex: bool = 'noindex' in robots_content
        nofollow: bool = 'nofollow' in robots_content # Checks if links on page should be followed

        # Determine effective directives (assuming 'index, follow' if tag missing)
        effective_index = 'noindex' if noindex else 'index'
        effective_follow = 'nofollow' if nofollow else 'follow'

        results['meta_robots'] = {
            'value': f"{effective_index}, {effective_follow}" + (f" (Tag: {robots_content})" if robots_content else " (Tag not found)"),
            'pass': not noindex, # Pass if it's indexable
            'details': f"Meta Robots directive: Allows indexing? {'No' if noindex else 'Yes'}. Allows following links? {'No' if nofollow else 'Yes'}."
            }
        if noindex: recommendations.append("CRITICAL: Page has a 'noindex' directive in the meta robots tag, preventing search engine indexing. Remove if unintentional.")
        if nofollow: recommendations.append("INFO: Page has a 'nofollow' directive in the meta robots tag. Links on this page may not pass equity. Verify if intentional.")

        # Canonical Tag
        canonical_tag = soup.find('link', rel='canonical')
        canonical_url: Optional[str] = canonical_tag.get('href', '').strip() if canonical_tag else None

        results['canonical_tag'] = {'value': canonical_url if canonical_url else 'Not specified', 'pass': bool(canonical_url), 'details': f"Canonical URL specified: {canonical_url if canonical_url else 'None Found'}"}

        # Check if canonical URL points to a significantly different URL
        # Normalize by removing scheme, www., and trailing slash for basic comparison
        def normalize_url(u: Optional[str]) -> Optional[str]:
            if not u: return None
            try:
                p = urlparse(u)
                # Remove scheme, www., path trailing slash
                netloc = p.netloc.replace('www.', '')
                path = p.path.rstrip('/') or '/' # Ensure root path is '/'
                return f"{netloc}{path}"
            except Exception:
                return u # Return original on parsing error

        norm_analyzed_url = normalize_url(url)
        norm_canonical_url = normalize_url(canonical_url)

        if not canonical_url:
             recommendations.append("Add Canonical Tag: Specify a self-referencing canonical tag (<link rel=\"canonical\" href=\"{full_url}\"/>) to consolidate signals and prevent potential duplicate content issues.")
        elif norm_canonical_url != norm_analyzed_url:
             recommendations.append(f"Review Canonical Tag: Canonical URL ({canonical_url}) appears different from the analyzed URL ({url}). Ensure this is intentional (e.g., for parameter handling or syndication).")

    except Exception as e:
        st.error(f"Error analyzing Crawlability/Indexability tags: {e}")
        results['robots_txt_check'] = results['meta_robots'] = results['canonical_tag'] = {'value': 'Error', 'pass': False, 'details': str(e)}


    return results, recommendations


def calculate_overall_score(on_page_results: ResultDict, tech_results: ResultDict) -> Dict[str, int]:
    """
    Calculates a weighted score based on analysis results (No PageSpeed).
    Weights are subjective and can be adjusted.

    Args:
        on_page_results: Dictionary of on-page analysis results.
        tech_results: Dictionary of technical analysis results.

    Returns:
        A dictionary with 'overall', 'on_page', and 'technical' scores (0-100).
    """
    # Define weights (adjust as needed, ensure reasonable distribution)
    # These weights are subjective examples.
    weights: Dict[str, float] = {
        # Category Weights
        'on_page': 0.6, 'technical': 0.4,

        # On-page sub-weights (sum should ideally reflect importance within category)
        'title': 0.18,          # Presence, Length, Keywords
        'meta_desc': 0.12,      # Presence, Length, Keywords
        'h1': 0.15,             # Presence (exactly 1), Keywords
        'subheadings': 0.05,    # Presence of H2-H6
        'content_length': 0.10, # Word count threshold
        'keywords': 0.08,       # Prominence (in first 100 words) - Density removed from score
        'readability': 0.07,    # Basic readability pass
        'images_alt': 0.10,     # Alt text presence
        'images_filename': 0.05,# Descriptive filenames
        'internal_links': 0.05, # Presence of internal links (if content long enough)
        'broken_links': 0.05,   # Absence of broken links

        # Technical sub-weights
        'https': 0.25,          # Presence of HTTPS
        'crawl_meta': 0.20,     # Meta robots (noindex check)
        'crawl_robots': 0.10,   # Robots.txt allowed check
        'canonical': 0.10,      # Canonical tag presence (self-referencing implicitly checked in recs)
        'url_structure': 0.10,  # Length, Readability
        'schema': 0.10,         # Schema presence
        'mobile_viewport': 0.10,# Viewport tag presence
        'ttfb': 0.05            # Basic TTFB threshold
    }

    scores: Dict[str, float] = {'on_page': 0.0, 'technical': 0.0}
    max_scores: Dict[str, float] = {'on_page': 0.0, 'technical': 0.0}

    def get_pass_score(result_dict: ResultDict, key: str) -> int:
        # Helper to safely get pass status (1 if True, 0 otherwise)
        # Returns 0 if key missing, pass is False, pass is None, or value is 'Error'
        item = result_dict.get(key, {})
        if item.get('value') == 'Error':
            return 0
        return 1 if item.get('pass', False) else 0

    # --- On-Page Scoring ---
    try:
        op_score: float = 0.0
        op_max: float = 0.0

        # Title (3 checks)
        op_max += weights['title']; op_score += weights['title'] * (
            0.4 * get_pass_score(on_page_results, 'title_presence') +
            0.3 * get_pass_score(on_page_results, 'title_length') +
            0.3 * get_pass_score(on_page_results, 'title_keywords')
            # Modifier check not included in score
        )
        # Meta Desc (3 checks)
        op_max += weights['meta_desc']; op_score += weights['meta_desc'] * (
            0.4 * get_pass_score(on_page_results, 'meta_desc_presence') +
            0.3 * get_pass_score(on_page_results, 'meta_desc_length') +
            0.3 * get_pass_score(on_page_results, 'meta_desc_keywords')
        )
        # H1 (2 checks)
        op_max += weights['h1']; op_score += weights['h1'] * (
            0.6 * get_pass_score(on_page_results, 'h1_presence') + # Crucial to have exactly one
            0.4 * get_pass_score(on_page_results, 'h1_keywords')
        )
        # Subheadings (1 check)
        op_max += weights['subheadings']; op_score += weights['subheadings'] * get_pass_score(on_page_results, 'h2_h6_structure')
        # Content Length (1 check)
        op_max += weights['content_length']; op_score += weights['content_length'] * get_pass_score(on_page_results, 'content_word_count')
        # Keyword Prominence (1 check)
        op_max += weights['keywords']; op_score += weights['keywords'] * get_pass_score(on_page_results, 'keyword_prominence')
        # Readability (1 check)
        op_max += weights['readability']; op_score += weights['readability'] * get_pass_score(on_page_results, 'readability')
        # Image Alt Text (1 check)
        op_max += weights['images_alt']; op_score += weights['images_alt'] * get_pass_score(on_page_results, 'image_alt_text')
        # Image Filenames (1 check)
        op_max += weights['images_filename']; op_score += weights['images_filename'] * get_pass_score(on_page_results, 'image_filenames')
        # Internal Links (1 check)
        op_max += weights['internal_links']; op_score += weights['internal_links'] * get_pass_score(on_page_results, 'internal_links_count')
        # Broken Links (1 check) - Pass = 0 broken links
        op_max += weights['broken_links']; op_score += weights['broken_links'] * get_pass_score(on_page_results, 'broken_links')


        scores['on_page'] = (op_score / op_max * 100) if op_max > 0 else 0.0
        max_scores['on_page'] = op_max # Store max possible score for debugging if needed

    except Exception as e:
        st.error(f"Error calculating On-Page score: {e}\n{traceback.format_exc()}")
        scores['on_page'] = 0.0


    # --- Technical Scoring ---
    try:
        tech_score: float = 0.0
        tech_max: float = 0.0

        # HTTPS (1 check)
        tech_max += weights['https']; tech_score += weights['https'] * get_pass_score(tech_results, 'https')
        # Meta Robots (1 check - noindex)
        tech_max += weights['crawl_meta']; tech_score += weights['crawl_meta'] * get_pass_score(tech_results, 'meta_robots')
        # Robots.txt (1 check - allowed)
        tech_max += weights['crawl_robots']; tech_score += weights['crawl_robots'] * get_pass_score(tech_results, 'robots_txt_check')
        # Canonical (1 check - presence)
        tech_max += weights['canonical']; tech_score += weights['canonical'] * get_pass_score(tech_results, 'canonical_tag')
        # URL Structure (Combined Length & Readability)
        tech_max += weights['url_structure']; tech_score += weights['url_structure'] * (
             0.5 * get_pass_score(tech_results, 'url_length') +
             0.5 * get_pass_score(tech_results, 'url_readability')
        )
        # Schema (1 check - presence)
        tech_max += weights['schema']; tech_score += weights['schema'] * get_pass_score(tech_results, 'schema_markup')
        # Mobile Viewport (1 check - presence)
        tech_max += weights['mobile_viewport']; tech_score += weights['mobile_viewport'] * get_pass_score(tech_results, 'mobile_viewport')
        # TTFB (1 check)
        tech_max += weights['ttfb']; tech_score += weights['ttfb'] * get_pass_score(tech_results, 'ttfb')

        scores['technical'] = (tech_score / tech_max * 100) if tech_max > 0 else 0.0
        max_scores['technical'] = tech_max

    except Exception as e:
        st.error(f"Error calculating Technical score: {e}\n{traceback.format_exc()}")
        scores['technical'] = 0.0

    # --- Overall Score ---
    overall_score: float = 0.0
    try:
        # Ensure scores are within 0-100 range before weighting
        scores['on_page'] = max(0.0, min(100.0, scores['on_page']))
        scores['technical'] = max(0.0, min(100.0, scores['technical']))

        overall_score = (scores['on_page'] * weights['on_page']) + (scores['technical'] * weights['technical'])
        # Ensure overall is also 0-100
        overall_score = max(0.0, min(100.0, overall_score))

    except Exception as e:
        st.error(f"Error calculating Overall score: {e}\n{traceback.format_exc()}")
        overall_score = 0.0

    # Return scores rounded to integers
    return {
        'overall': int(round(overall_score)),
        'on_page': int(round(scores['on_page'])),
        'technical': int(round(scores['technical']))
    }


def generate_recommendations(
    on_page_recs: List[str],
    tech_recs: List[str],
    on_page_results: ResultDict, # Pass results for context if needed
    tech_results: ResultDict
) -> List[str]:
    """
    Combines, prioritizes, and formats recommendations based on severity.

    Args:
        on_page_recs: List of recommendations from on-page analysis.
        tech_recs: List of recommendations from technical analysis.
        on_page_results: Dictionary of on-page results (for context).
        tech_results: Dictionary of technical results (for context).

    Returns:
        A sorted list of prioritized recommendation strings (max 15).
    """
    prioritized: List[Tuple[int, str]] = [] # Store as (priority_level, message)
    processed_recs: set[str] = set() # Keep track of unique recommendation messages

    # Priority levels (lower number = higher priority)
    PRIORITY_CRITICAL = 1
    PRIORITY_HIGH = 2
    PRIORITY_MEDIUM = 3
    PRIORITY_LOW = 4
    PRIORITY_INFO = 5

    def add_rec(rec_message: str, priority_level: int):
        # Extract the core message after the initial prefix (if any)
        core_message = rec_message.split(': ', 1)[-1] if ': ' in rec_message else rec_message
        core_message = core_message.strip()

        if core_message and core_message not in processed_recs:
            prioritized.append((priority_level, core_message))
            processed_recs.add(core_message)

    # Combine all generated recommendations
    all_recs = tech_recs + on_page_recs

    # Process recommendations and assign priorities
    for rec in all_recs:
        rec_lower = rec.lower()
        priority = PRIORITY_LOW # Default

        # Assign Priority based on keywords or explicit markers
        if "critical" in rec_lower: priority = PRIORITY_CRITICAL
        elif any(kw in rec_lower for kw in ["high:", "broken links", "exactly one h1", "add a compelling title", "add a unique meta description", "add an h1 tag"]): priority = PRIORITY_HIGH
        elif any(kw in rec_lower for kw in ["medium:", "improve readability", "improve url", "server response time", "review canonical", "internal linking", "alt text", "image seo", "mobile experience", "add canonical tag", "implement schema markup", "improve content depth"]): priority = PRIORITY_MEDIUM
        elif "info:" in rec_lower: priority = PRIORITY_INFO
        elif "low:" in rec_lower: priority = PRIORITY_LOW # Explicit low

        add_rec(rec, priority)

    # Ensure certain manual checks are always mentioned if not already covered by specific errors
    manual_checks = {
        "Review Page Speed Manually: Use tools like Google PageSpeed Insights (web.dev/measure) for Core Web Vitals.": PRIORITY_MEDIUM,
        "Manually verify mobile font sizes, tap targets, and check for intrusive interstitials.": PRIORITY_MEDIUM,
        "Manually review image compression, formats (WebP/AVIF), and dimensions.": PRIORITY_LOW,
        "Manually review content quality, accuracy, and alignment with search intent.": PRIORITY_HIGH, # Content quality is crucial
    }
    for check, level in manual_checks.items():
        add_rec(check, level)


    # Sort by priority level (ascending) then alphabetically
    prioritized.sort(key=lambda x: (x[0], x[1]))

    # Format with priority prefixes
    priority_map = {
        PRIORITY_CRITICAL: "🔴 CRITICAL",
        PRIORITY_HIGH:     "🟠 HIGH",
        PRIORITY_MEDIUM:   "🔵 MEDIUM",
        PRIORITY_LOW:      "⚪️ LOW",
        PRIORITY_INFO:     "ℹ️ INFO"
    }
    formatted_recs = [f"{priority_map.get(level, '⚪️ LOW')}: {message}" for level, message in prioritized]

    return formatted_recs[:15] # Limit to top 15 recommendations


# --- Visualization Functions ---

def create_score_gauge(score: Optional[Union[int, float]], title: str) -> alt.Chart:
    """
    Creates a semi-circle gauge visualization using Altair (v5+ compatible).

    Args:
        score: The score value (0-100).
        title: The title for the gauge chart.

    Returns:
        An Altair Chart object representing the gauge.
    """
    if score is None: score = 0
    # Ensure score is within 0-100
    score = max(0, min(100, int(score)))

    # Define color scale for the score arc and text
    color_scale = alt.Scale(
        domain=[0, 50, 90, 100], # Score thresholds for colors
        range=['#F44336', '#FFC107', '#4CAF50', '#4CAF50'] # Red, Yellow, Green
    )

    # Base background arc (full semi-circle)
    # Data needs to be structured for theta to map correctly to angle (0 to pi)
    base_data = pd.DataFrame({'value': [100], 'order': [1]}) # Full range, draw first
    base = alt.Chart(base_data).mark_arc(outerRadius=100, innerRadius=80).encode(
        # Use theta2 to define the end angle (pi for semi-circle)
        theta=alt.Theta('value', scale=alt.Scale(domain=[0, 100], range=[0, 3.14159265]), stack=None), # Map 100 to Pi
        color=alt.value('#e0e0e0'), # Light grey background
        order=alt.Order('order', sort='descending') # Ensure it's drawn behind score arc
    )

    # Foreground score arc
    score_data = pd.DataFrame({'value': [score], 'order': [0]}) # Score value, draw second
    score_arc = alt.Chart(score_data).mark_arc(outerRadius=100, innerRadius=80).encode(
        # Map the score value (0-100) to the angle range (0 to Pi)
        # Theta maps the 'value' field to the angle. Scale defines the mapping.
        theta=alt.Theta("value", scale=alt.Scale(domain=[0, 100], range=[0, 3.14159265]), stack=None), # REMOVED legend=None
        color=alt.Color("value", scale=color_scale, legend=None), # Color based on score value
        order=alt.Order('order', sort='descending') # Draw score arc on top
    )

    # Score text in the middle
    text = alt.Chart(pd.DataFrame({'value': [score]})).mark_text(
        align='center',
        baseline='middle', # Vertically center
        dy=-15, # Adjust vertical position slightly upward from center of arc
        fontSize=30,
        fontWeight='bold'
    ).encode(
        text=alt.Text("value", format=".0f"), # Display score as integer
        color=alt.Color("value", scale=color_scale, legend=None) # Color text based on score
    )

    # Layer the charts
    chart = alt.layer(base, score_arc, text).properties(
        title=alt.TitleParams(text=title, anchor='middle', dy=-80, fontSize=14) # Center title, move further up
    ).configure_view(
        strokeWidth=0 # Remove border around the chart view
    ).configure_axis(
        grid=False, # Remove grid lines if any appear
        domain=False # Remove axis domain lines
    )
    return chart


def create_comparison_bar_chart(df_compare: pd.DataFrame, metric: str, title: str) -> Optional[alt.Chart]:
    """
    Creates a grouped bar chart for comparing target vs competitors on a specific metric.

    Args:
        df_compare: DataFrame containing comparison data (must include 'Competitor' and metric column).
        metric: The column name of the metric to plot.
        title: The title for the bar chart.

    Returns:
        An Altair Chart object or None if data is invalid or empty.
    """
    if metric not in df_compare.columns:
        st.caption(f"Metric '{metric}' not available for comparison chart.")
        return None

    # Attempt to clean and convert metric column to numeric
    try:
        # Handle potential non-numeric placeholders like 'N/A', 'Error', 'Pending check...'
        df_compare[metric] = df_compare[metric].replace(['N/A', 'Error', 'Pending check...', None, ''], pd.NA)

        # Handle boolean-like strings ('Yes'/'No') if relevant for the metric
        if df_compare[metric].astype(str).str.contains('Yes|No', case=False, na=False).any():
             df_compare[metric] = df_compare[metric].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}, regex=False)

        # Convert to numeric, coercing errors to NaN
        df_compare[metric] = pd.to_numeric(df_compare[metric], errors='coerce')

        # Drop rows where the metric is now NaN after coercion
        df_compare_clean = df_compare.dropna(subset=[metric]).copy() # Work on a copy

        # Ensure the metric column is indeed numeric after cleaning
        if not pd.api.types.is_numeric_dtype(df_compare_clean[metric]):
            st.caption(f"Metric '{metric}' could not be converted to a numeric type for charting.")
            return None

    except Exception as e:
        st.caption(f"Could not process metric '{metric}' for chart: {e}")
        return None

    if df_compare_clean.empty:
        st.caption(f"Not enough valid data to display {title} chart for metric '{metric}'.")
        return None

    # Create the bar chart
    try:
        # Use the cleaned DataFrame
        chart = alt.Chart(df_compare_clean).mark_bar().encode(
            # Sort bars by the metric value (descending)
            x=alt.X('Competitor', sort='-y', title=None, axis=alt.Axis(labelAngle=-45)), # Angle labels if needed
            y=alt.Y(metric, title=metric), # Use the metric name as Y-axis title
            color=alt.Color('Competitor', legend=alt.Legend(title="Site", orient="top")), # Add legend title
            tooltip=['Competitor', alt.Tooltip(metric, title=metric, format=".0f")] # Format tooltip number
        ).properties(
            title=title
        ).interactive() # Allow zooming and panning
        return chart
    except Exception as e:
        st.error(f"Error creating comparison chart for '{metric}': {e}")
        return None


# --- Main Streamlit App ---

def main():
    st.set_page_config(page_title="Single Page SEO Analyzer", layout="wide", initial_sidebar_state="expanded")

    st.title("📄 Single Page SEO Analyzer & Competitor Benchmark")
    st.markdown("""
        Analyze On-Page and Technical SEO factors for a single URL.
        **Excludes:** Google PageSpeed Insights / Core Web Vitals (requires manual checks).
        **Competitor analysis:** Uses simulated data by default. Enable real data by setting
        `SERP_API_KEY` and `SERP_API_ENDPOINT` secrets in Streamlit Cloud and uncommenting the API call block.
    """)

    # --- Inputs ---
    st.sidebar.header("Inputs")
    url_input: str = st.sidebar.text_input("Target Webpage URL", placeholder="https://www.example.com/page")
    keywords_input: str = st.sidebar.text_area("Target Keywords (one per line, main keyword first)", placeholder="best coffee grinder\nburr grinder review")
    run_analysis: bool = st.sidebar.button("Analyze Page", type="primary")

    # --- Limitations and API Key Info ---
    with st.expander("Important Notes & Limitations", expanded=False):
        st.info("""
            * **Single Page Focus:** Analyzes ONLY the provided URL's static HTML content.
            * **No Page Speed / CWV:** Does **NOT** include Google PageSpeed Insights (Core Web Vitals). Manual checks using [PageSpeed Insights](https://pagespeed.web.dev/) or [web.dev/measure](https://web.dev/measure/) are essential. Basic TTFB (Time To First Byte) is shown as an indicator.
            * **Competitor Data (Simulated/Optional API):** Real competitor data requires a paid SERP API. Uses placeholder data unless SERP API secrets are configured *and* the API call code block inside `get_serp_competitors_placeholder` is uncommented and adapted for your provider.
            * **Dynamic Content:** May not fully capture content rendered heavily by JavaScript. Use Browser DevTools 'Inspect Element' on the live page for JS-heavy sites.
            * **Qualitative Aspects:** Content quality, user experience (UX), E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness), and true search intent matching require manual review.
            * **Robots.txt Parsing:** Uses a basic parser; complex rules or non-standard directives might be misinterpreted.
            * **Broken Link Check:** Checks reachability at analysis time; links might change later. Uses HEAD then GET requests.
        """)
        if not SERP_API_KEY or not SERP_API_ENDPOINT:
            st.warning("⚠️ **Note:** Real-time competitor analysis requires SERP API secrets set in Streamlit Cloud and code modification. Currently using placeholder competitor data.")
        else:
            st.success("✅ SERP API credentials found (using Secrets/Env Vars). Competitor analysis will attempt to use the API if the relevant code block is uncommented.")

    # --- Analysis Execution ---
    if run_analysis and url_input and keywords_input:
        if not is_valid_url(url_input):
            st.error("Invalid URL format. Please enter a full URL starting with http:// or https://")
        else:
            keywords_list: List[str] = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]
            if not keywords_list:
                st.error("Please enter at least one target keyword.")
            else:
                st.success(f"Analyzing **{url_input}** for keywords: **{', '.join(keywords_list)}**")
                analysis_placeholder = st.empty() # Placeholder for status updates
                analysis_placeholder.info("🚀 Starting analysis... Fetching target page...")

                # --- Initialize Results ---
                on_page_results: ResultDict = {}
                tech_results: ResultDict = {}
                competitor_data: pd.DataFrame = pd.DataFrame()
                serp_features: Dict[str, Any] = {}
                on_page_recommendations: List[str] = []
                tech_recommendations: List[str] = []
                final_recommendations: List[str] = []
                scores: Dict[str, int] = {'overall': 0, 'on_page': 0, 'technical': 0}
                all_links_on_page: List[str] = []
                broken_links_result: List[Tuple[str, str]] = []
                target_domain: Optional[str] = get_domain(url_input)

                # Fetch Target URL HTML
                html_content, status_code, ttfb, fetch_error = fetch_html(url_input)

                if fetch_error or not html_content:
                    analysis_placeholder.error(f"Analysis Stopped: Failed to fetch target URL ({status_code if status_code else 'N/A'}): {fetch_error}")
                    st.stop() # Stop execution if fetch fails critically

                try:
                    analysis_placeholder.info("⚙️ Parsing HTML and analyzing On-Page SEO...")
                    soup = BeautifulSoup(html_content, 'lxml')

                    # Run On-Page Analysis
                    on_page_results, on_page_recommendations, all_links_on_page = analyze_on_page(soup, url_input, keywords_list, html_content)

                    # Run Technical Analysis
                    analysis_placeholder.info("🛠️ Analyzing Technical SEO...")
                    tech_results, tech_recommendations = analyze_technical(soup, url_input, html_content, ttfb, keywords_list)

                    # --- Run External Checks ---
                    # Robots.txt
                    analysis_placeholder.info("🤖 Checking Robots.txt rules...")
                    robots_txt_content, robots_error = fetch_robots_txt(url_input)

                    if robots_error and "Assuming allowed" not in robots_error: # Handle fetch errors
                         tech_results['robots_txt_check'] = {'value': 'Error Fetching', 'pass': False, 'details': robots_error}
                         tech_recommendations.append(f"Warning: Could not verify robots.txt: {robots_error}")
                    else: # Process content (even if empty/4xx)
                         url_path: str = urlparse(url_input).path or "/"
                         is_blocked: bool = is_disallowed(robots_txt_content, url_path, USER_AGENT) # Check for our specific UA
                         pass_status = not is_blocked
                         details_msg = f"Path '{url_path}' appears {'**blocked**' if is_blocked else 'allowed'} by robots.txt for user-agent '{USER_AGENT}'."
                         if robots_error: # Add the "Assuming allowed" message if it occurred
                             details_msg += f" ({robots_error})"
                         tech_results['robots_txt_check'] = {'value': 'Blocked' if is_blocked else 'Allowed', 'pass': pass_status, 'details': details_msg}
                         if is_blocked:
                             tech_recommendations.append(f"CRITICAL: Page ('{url_path}') might be blocked by robots.txt for '{USER_AGENT}'. Verify rules.")
                         elif robots_txt_content is None and not robots_error: # Should not happen based on fetch_robots_txt logic, but safeguard
                             tech_results['robots_txt_check'] = {'value': 'Error Processing', 'pass': False, 'details': 'Failed to process robots.txt content.'}


                    # Broken Links Check (can be slow)
                    analysis_placeholder.info("🔗 Checking for Broken Links (this may take a moment)...")
                    if all_links_on_page:
                        broken_links_result = check_broken_links(all_links_on_page)
                        num_broken: int = len(broken_links_result)
                        # Update results placeholder
                        if 'broken_links' in on_page_results:
                            on_page_results['broken_links'] = {
                                'value': f"{num_broken} found",
                                'pass': num_broken == 0,
                                'details': f"Found {num_broken} broken or problematic links (4xx/5xx/Timeout/Error)."
                                }
                            if num_broken > 0:
                                on_page_recommendations.append(f"HIGH: Fix Broken Links: Found {num_broken} potentially broken links. See details below.")
                        else: # Should not happen if analyze_on_page ran ok
                             st.warning("Could not update broken link status in results.")
                    else:
                         if 'broken_links' in on_page_results:
                              on_page_results['broken_links'] = {'value': '0 found', 'pass': True, 'details': 'No outgoing links found to check.'}


                    # Competitor Analysis (Placeholder or Real if API Keys Set)
                    analysis_placeholder.info("📈 Analyzing Competitors (Simulated or via SERP API)...")
                    competitors_list, serp_features, serp_error = get_serp_competitors_placeholder(keywords_list, target_domain or "") # Pass empty string if None
                    if serp_error and "credentials not configured" not in serp_error: # Show error only if keys ARE set but API failed
                        st.error(f"Competitor analysis failed: {serp_error}")

                    competitor_analysis_results: List[Dict[str, Any]] = []
                    if competitors_list:
                        # In a real scenario, you would fetch and analyze each competitor URL here.
                        # This requires uncommenting/implementing the fetch logic in the placeholder function
                        # or adding a separate loop here. For demo, we only use basic SERP data.
                        for comp in competitors_list:
                             comp_analysis = {
                                 'Competitor': comp.get('domain', 'N/A'),
                                 'URL': comp.get('link'),
                                 'SERP Title': comp.get('title','N/A')[:100], # From SERP
                                 'Title Length': len(comp.get('title','')), # From SERP Title
                                 # --- Metrics below REQUIRE fetching/analyzing each competitor page ---
                                 # Add placeholders or fetch real data if implemented
                                 # 'Word Count': 'N/A',
                                 # 'Images': 'N/A',
                                 # 'Internal Links': 'N/A',
                                 # 'Schema Markup?': 'N/A'
                             }
                             competitor_analysis_results.append(comp_analysis)

                        competitor_data = pd.DataFrame(competitor_analysis_results)

                    # Calculate Scores & Final Recommendations
                    analysis_placeholder.info("💯 Calculating scores and finalizing recommendations...")
                    # Pass updated results including broken link status
                    scores = calculate_overall_score(on_page_results, tech_results)
                    final_recommendations = generate_recommendations(on_page_recommendations, tech_recommendations, on_page_results, tech_results)

                    analysis_placeholder.empty() # Clear status message
                    st.success("✅ Analysis Complete!")
                    st.markdown("---") # Separator

                    # --- Display Results ---
                    st.header("📊 SEO Analysis Results")

                    # Overall Score Gauges
                    st.subheader("Overall Performance Score")
                    score_col1, score_col2, score_col3 = st.columns(3)
                    with score_col1:
                        gauge_overall = create_score_gauge(scores.get('overall'), "Overall Score")
                        st.altair_chart(gauge_overall, use_container_width=True)
                    with score_col2:
                        gauge_onpage = create_score_gauge(scores.get('on_page'), "On-Page Score")
                        st.altair_chart(gauge_onpage, use_container_width=True)
                    with score_col3:
                        gauge_tech = create_score_gauge(scores.get('technical'), "Technical Score")
                        st.altair_chart(gauge_tech, use_container_width=True)

                    # Prioritized Recommendations
                    st.subheader("💡 Prioritized Recommendations (Top 15)")
                    if final_recommendations:
                        for i, rec in enumerate(final_recommendations):
                            st.markdown(f"{i+1}. {rec}")
                    else:
                        st.success("✅ No major recommendations identified based on checks!")

                    # Weaknesses & Opportunities (simplified)
                    st.subheader("📉 Strengths, Weaknesses & Opportunities")
                    weaknesses = [rec for rec in final_recommendations if "🔴 CRITICAL" in rec or "🟠 HIGH" in rec]
                    opportunities = [rec for rec in final_recommendations if "🔵 MEDIUM" in rec or "⚪️ LOW" in rec]
                    # Identify strengths (passed checks) - basic example
                    strengths = []
                    for key, data in {**on_page_results, **tech_results}.items():
                         # Focus on key positive indicators
                        is_strength = False
                        if key == 'https' and data.get('pass'): is_strength = True
                        if key == 'meta_robots' and data.get('pass'): is_strength = True # i.e., indexable
                        if key == 'title_presence' and data.get('pass'): is_strength = True
                        if key == 'h1_presence' and data.get('pass'): is_strength = True
                        if key == 'mobile_viewport' and data.get('pass'): is_strength = True

                        if is_strength:
                            param_name = key.replace('_', ' ').title()
                            strengths.append(f"{param_name}") # Simple list of passed key checks


                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**👍 Key Strengths:**")
                        if strengths:
                            st.markdown("\n".join(f"- ✅ {s}" for s in strengths[:5])) # Show top 5
                        else: st.write("Basic checks passed.")
                    with col2:
                        st.markdown("**👎 Key Weaknesses:**")
                        if weaknesses:
                             # Show message part without prefix
                             st.markdown("\n".join(f"- {w.split(': ', 1)[-1]}" for w in weaknesses))
                        else: st.write("No critical/high issues found.")
                    with col3:
                         st.markdown("**✨ Opportunities:**")
                         if opportunities:
                             # Show message part without prefix
                             st.markdown("\n".join(f"- {o.split(': ', 1)[-1]}" for o in opportunities[:5])) # Show top 5 medium/low

                         # Add content gap / SERP feature opportunities if SERP API was real and returned data
                         if serp_features and serp_features.get('related_questions'):
                             st.markdown("**Content Ideas (from PAA):**")
                             for q in serp_features.get('related_questions', [])[:3]: # Show top 3 PAA
                                 st.markdown(f"- {q.get('question', 'N/A')}")
                         elif not opportunities:
                             st.write("No specific low/medium priority actions identified.")


                    st.markdown("---")
                    # Detailed Analysis Tabs
                    st.subheader("🔍 Detailed Analysis Checklist")
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "On-Page SEO",
                        "Technical SEO",
                        "Broken Links",
                        "Competitor Analysis"
                        ])

                    # Helper function to create status string for DataFrames
                    def get_status_string(data_dict: ResultDict, key: str) -> str:
                        item = data_dict.get(key, {})
                        pass_status = item.get('pass')
                        value = item.get('value')
                        details = item.get('details', '')

                        if value == 'Error': return '❌ Error'
                        if value == 'Pending check...': return '⏳ Pending'
                        if pass_status is True: return '✅ Pass'
                        if pass_status is False:
                            # Check if it's a critical issue based on common terms
                            if any(crit in str(value).upper() or crit in details.upper() for crit in ['CRITICAL', 'NOINDEX', 'BLOCKED', 'HTTPS: NO']):
                                return '❌ CRITICAL'
                            return '⚠️ Needs Improvement'
                        return 'ℹ️ Info / Manual Check' # If pass is None or not applicable


                    with tab1: # On-Page
                        st.subheader("On-Page SEO Details")
                        on_page_list = []
                        # Define display order if desired
                        on_page_order = [
                            'title_presence', 'title_length', 'title_keywords', 'title_modifiers',
                            'meta_desc_presence', 'meta_desc_length', 'meta_desc_keywords',
                            'h1_presence', 'h1_keywords', 'h1_content', 'h2_h6_structure',
                            'content_word_count', 'keyword_prominence', 'keyword_density', 'readability',
                            'image_count', 'image_alt_text', 'image_filenames',
                            'internal_links_count', 'external_links_count', 'internal_anchors', 'external_anchors',
                            'broken_links' # Placeholder status
                        ]
                        displayed_keys = set()
                        for key in on_page_order:
                             if key in on_page_results:
                                data = on_page_results[key]
                                param_name = key.replace('_', ' ').title()
                                on_page_list.append({'Parameter': param_name, 'Result / Details': data.get('details', 'N/A'), 'Status': get_status_string(on_page_results, key)})
                                displayed_keys.add(key)
                        # Add any remaining keys not in the defined order
                        for key, data in on_page_results.items():
                            if key not in displayed_keys:
                                param_name = key.replace('_', ' ').title()
                                on_page_list.append({'Parameter': param_name, 'Result / Details': data.get('details', 'N/A'), 'Status': get_status_string(on_page_results, key)})

                        if on_page_list:
                            on_page_df = pd.DataFrame(on_page_list)
                            st.dataframe(on_page_df, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No on-page analysis results to display.")

                    with tab2: # Technical
                        st.subheader("Technical SEO Details")
                        tech_list = []
                        # Define display order
                        tech_order = [
                            'https', 'ttfb', 'robots_txt_check', 'meta_robots', 'canonical_tag',
                            'mobile_viewport', 'mobile_fonts', 'mobile_interstitials', 'mobile_tap_targets',
                            'url_length', 'url_keywords', 'url_readability', 'schema_markup'
                        ]
                        displayed_keys = set()
                        for key in tech_order:
                            if key in tech_results:
                                data = tech_results[key]
                                param_name = key.replace('_', ' ').title()
                                tech_list.append({'Parameter': param_name, 'Result / Details': data.get('details', 'N/A'), 'Status': get_status_string(tech_results, key)})
                                displayed_keys.add(key)
                        # Add any remaining keys
                        for key, data in tech_results.items():
                             if key not in displayed_keys:
                                param_name = key.replace('_', ' ').title()
                                tech_list.append({'Parameter': param_name, 'Result / Details': data.get('details', 'N/A'), 'Status': get_status_string(tech_results, key)})

                        if tech_list:
                            tech_df = pd.DataFrame(tech_list)
                            st.dataframe(tech_df, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No technical analysis results to display.")


                    with tab3: # Broken Links Detail
                         st.subheader("Detected Broken Links / Errors")
                         if broken_links_result:
                             broken_df = pd.DataFrame(broken_links_result, columns=['URL', 'Status / Error'])
                             st.dataframe(broken_df, use_container_width=True, hide_index=True)
                             st.markdown(f"Found **{len(broken_links_result)}** potentially broken or problematic links.")
                         else:
                             st.info("No broken links detected or no links found to check.")

                    with tab4: # Competitor
                        st.subheader("Competitor Benchmark Report")
                        if not competitor_data.empty:
                            st.markdown("**Top Simulated/Identified Competitors (from SERP):**")
                            # Display basic competitor info from SERP list
                            comp_display_df = competitor_data[['Competitor', 'URL', 'SERP Title']].copy()
                            comp_display_df.index = range(1, len(comp_display_df) + 1) # Start index at 1
                            st.dataframe(comp_display_df, use_container_width=True)


                            st.markdown("**Comparative Analysis (Basic - Requires fetching competitor pages for full detail):**")

                            # Prepare data for comparison chart - Add target page data
                            target_metrics = {
                                'Competitor': f"⭐ Target Page", # Mark target page
                                'URL': url_input,
                                'SERP Title': on_page_results.get('title_presence',{}).get('details', 'N/A')[:100], # Use actual Title
                                'Title Length': on_page_results.get('title_length',{}).get('value'),
                                'Word Count': on_page_results.get('content_word_count',{}).get('value'),
                                'Images': on_page_results.get('image_count',{}).get('value'),
                                'Internal Links': on_page_results.get('internal_links_count',{}).get('value'),
                                'Schema Markup?': 'Yes' if tech_results.get('schema_markup', {}).get('pass') else 'No'
                                # Add more metrics here if they are calculated for target page
                            }

                            # Combine target and competitor data
                            compare_list = [target_metrics] + competitor_data.to_dict('records')
                            compare_df = pd.DataFrame(compare_list)

                            # Define columns to show in comparison table - adjust if fetching more competitor data
                            display_cols = ['Competitor', 'Title Length'] # Start with basic info from SERP/Target
                            # Add other columns IF they exist in the DataFrame (i.e., if fetched/analyzed)
                            for col in ['Word Count', 'Images', 'Internal Links', 'Schema Markup?']:
                                if col in compare_df.columns and compare_df[col].notna().any(): # Check if col exists AND has non-NA values
                                    # Ensure the column intended for numeric chart has actual numbers
                                    if col in ['Word Count', 'Images', 'Internal Links', 'Title Length']:
                                         compare_df[col] = pd.to_numeric(compare_df[col], errors='coerce')
                                    if compare_df[col].notna().any(): # Check again after coercion
                                         display_cols.append(col)

                            # Display comparison table (use Competitor as index)
                            st.dataframe(compare_df.set_index('Competitor')[display_cols], use_container_width=True)


                            # Add comparison charts only if numeric data exists
                            st.markdown("**Visual Comparison:**")
                            chart_metrics = ['Word Count', 'Internal Links', 'Title Length', 'Images']
                            chart_cols = st.columns(len([m for m in chart_metrics if m in display_cols]))
                            chart_idx = 0
                            for metric in chart_metrics:
                                if metric in display_cols:
                                    with chart_cols[chart_idx]:
                                        chart_comp = create_comparison_bar_chart(compare_df.copy(), metric, f'{metric} Comparison') # Pass copy
                                        if chart_comp:
                                            st.altair_chart(chart_comp, use_container_width=True)
                                        else:
                                            st.caption(f"Could not generate chart for {metric}.")
                                    chart_idx += 1
                            if chart_idx == 0:
                                st.info("No numeric metrics available for comparison charts.")


                            st.markdown("**SERP Features Analysis (Simulated or via SERP API):**")
                            if serp_features:
                                features_found_list = []
                                if serp_features.get('featured_snippet'): features_found_list.append("Featured Snippet")
                                if serp_features.get('related_questions'): features_found_list.append("People Also Ask")
                                if serp_features.get('knowledge_graph'): features_found_list.append("Knowledge Graph")
                                if serp_features.get('inline_videos'): features_found_list.append("Video Pack")
                                if serp_features.get('top_stories'): features_found_list.append("Top Stories")
                                if serp_features.get('local_pack'): features_found_list.append("Local Pack")
                                # Add checks for other features your API might provide

                                if features_found_list:
                                    st.write(f"- Features Detected on SERP: {', '.join(features_found_list)}")
                                    st.write("  Consider if your content could target these features (e.g., answer questions directly for PAA/Featured Snippets).")
                                else:
                                    st.write("- No major SERP features detected in simulation/API response for the primary keyword.")
                            else:
                                st.write("SERP Feature data not available (requires API call).")

                        else:
                            st.warning("Competitor data could not be generated or analyzed. Check SERP API configuration or the placeholder function.")


                except Exception as e:
                    analysis_placeholder.error(f"An unexpected error occurred during the main analysis process: {e}")
                    st.error(f"Traceback:\n{traceback.format_exc()}") # Show full traceback for debugging

    # Add footer or additional info if desired
    st.sidebar.markdown("---")
    st.sidebar.info("App Version: 2025-04-28 | SEO analysis requires manual interpretation and deeper dives.")


if __name__ == "__main__":
    main()
