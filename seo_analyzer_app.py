# seo_analyzer_app.py
# Streamlit application for Single Page SEO Analysis (No PageSpeed Insights)

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import altair as alt
from urllib.parse import urlparse, urljoin
import json
import time
import os
from readability import Document # Uses readability-lxml
import re
import validators # For URL validation
import traceback # For detailed error logging if needed

# --- Configuration & Constants ---
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
REQUEST_TIMEOUT = 15 # seconds

# SEO Best Practice Thresholds
TITLE_MAX_LEN = 60
META_DESC_MAX_LEN = 160
MIN_FONT_SIZE_PX = 16 # General recommendation - Basic Check Only Now

# API Keys from Environment Variables (SERP API only now)
# For Streamlit Cloud deployment, set these as Secrets in the app settings
SERP_API_KEY = os.environ.get('SERP_API_KEY') or st.secrets.get("SERP_API_KEY")
SERP_API_ENDPOINT = os.environ.get('SERP_API_ENDPOINT') or st.secrets.get("SERP_API_ENDPOINT")

# --- Helper Functions ---

def is_valid_url(url):
    """Validate the URL format."""
    try:
        return validators.url(url)
    except Exception:
        return False

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_html(url):
    """Fetches HTML content of a URL."""
    headers = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        # Try to detect encoding, fall back to apparent_encoding or utf-8
        response.encoding = response.apparent_encoding if response.encoding == 'ISO-8859-1' else response.encoding or 'utf-8'

        # Basic TTFB timing (Not a replacement for CWV!)
        ttfb = response.elapsed.total_seconds()
        return response.text, response.status_code, ttfb, None # content, status_code, ttfb, error
    except requests.exceptions.Timeout:
        return None, None, None, f"Request timed out after {REQUEST_TIMEOUT} seconds."
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching {url}: {e}") # Show network errors more prominently
        return None, getattr(e.response, 'status_code', None), None, f"Could not fetch URL: {e}"
    except Exception as e:
        return None, None, None, f"An unexpected error occurred during fetch: {e}"

def get_domain(url):
    """Extracts the domain name from a URL."""
    try:
        return urlparse(url).netloc
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_robots_txt(url):
    """Fetches and parses robots.txt for the domain."""
    domain = get_domain(url)
    if not domain:
        return None, "Invalid URL provided."
    # Construct robots.txt URL carefully (handle potential ports, ensure https)
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme or 'https'
    robots_url = f"{scheme}://{domain}/robots.txt"

    headers = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(robots_url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            # Only return content if successful
            return response.text, None
        elif response.status_code >= 400 and response.status_code < 500:
             # Treat 4xx errors (like 404 Not Found, 403 Forbidden) as permissive (allow crawling)
            return "", f"Robots.txt not found or inaccessible (Status: {response.status_code}). Assuming allowed."
        else:
            # Handle other errors (e.g., 5xx server errors)
            return None, f"Could not fetch robots.txt (Status: {response.status_code})."
    except requests.exceptions.RequestException as e:
        return None, f"Could not fetch robots.txt: {e}"
    except Exception as e:
         return None, f"Unexpected error fetching robots.txt: {e}"

def is_disallowed(robots_content, target_url_path, user_agent=USER_AGENT):
    """
    Checks if a specific path is disallowed by robots.txt content.
    Focuses on rules for '*' and the specified user_agent.
    Handles basic wildcards (*) and end-of-string ($).
    """
    if robots_content is None or not target_url_path:
        return False  # Assume allowed if no robots.txt or path

    # Ensure path starts with /
    if not target_url_path.startswith('/'):
        target_url_path = '/' + target_url_path

    lines = robots_content.splitlines()
    relevant_rules = {'*': {'allow': [], 'disallow': []}}
    specific_rules = {user_agent.lower(): {'allow': [], 'disallow': []}}
    current_agents = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        try:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == 'user-agent':
                agent = value.lower()
                current_agents = []
                if agent == '*':
                    current_agents.append('*')
                if agent == user_agent.lower():
                     # Add specific agent if not already covered by '*' or previous specific line
                     if agent not in specific_rules:
                         specific_rules[agent] = {'allow': [], 'disallow': []}
                     current_agents.append(agent)

            elif key in ['allow', 'disallow'] and value:
                 rule_path = value
                 # Store rules for current user agents
                 for agent in current_agents:
                      if agent == '*':
                          if rule_path not in relevant_rules['*'][key]:
                             relevant_rules['*'][key].append(rule_path)
                      elif agent in specific_rules:
                          if rule_path not in specific_rules[agent][key]:
                              specific_rules[agent][key].append(rule_path)

        except ValueError:
            continue # Ignore lines that don't split correctly

    # Determine rules applying to our specific user agent (specific rules override '*')
    final_rules = {
        'allow': relevant_rules['*']['allow'] + specific_rules.get(user_agent.lower(), {}).get('allow', []),
        'disallow': relevant_rules['*']['disallow'] + specific_rules.get(user_agent.lower(), {}).get('disallow', [])
    }

    # Check rules: Find the most specific matching rule
    longest_allow_match_len = -1
    longest_disallow_match_len = -1

    # Check Disallow rules
    for rule in final_rules['disallow']:
        pattern = re.escape(rule).replace(r'\*', '.*')
        if rule.endswith('$') and pattern.endswith(r'\$'):
             pattern = pattern[:-2] + '$' # Match end of string
        elif not rule.endswith('/'):
             # Ensure directory match works if rule doesn't end with / but path does
             # Example: rule /private matches /private/page.html
             pass # Regex will handle this partial match naturally

        try:
            if re.match(pattern, target_url_path):
                if len(rule) > longest_disallow_match_len:
                    longest_disallow_match_len = len(rule)
        except re.error:
            continue # Ignore invalid regex patterns

    # Check Allow rules
    for rule in final_rules['allow']:
        pattern = re.escape(rule).replace(r'\*', '.*')
        if rule.endswith('$') and pattern.endswith(r'\$'):
             pattern = pattern[:-2] + '$'
        try:
            if re.match(pattern, target_url_path):
                 if len(rule) > longest_allow_match_len:
                    longest_allow_match_len = len(rule)
        except re.error:
            continue

    # Disallowed if a disallow rule matches and no *more specific* allow rule matches
    if longest_disallow_match_len != -1 and longest_allow_match_len < longest_disallow_match_len:
        return True

    return False # Allowed by default or by a more specific allow rule


@st.cache_data(ttl=86400) # Cache for a day
def check_broken_links(links_to_check):
    """Checks a list of links for non-200 status codes using HEAD/GET requests."""
    broken = []
    # Ensure links are unique and absolute before checking
    unique_links = list(set(l for l in links_to_check if isinstance(l, str) and l.startswith('http')))

    if not unique_links:
        return []

    headers = {'User-Agent': USER_AGENT}
    # Use a session for potential connection reuse and better handling
    session = requests.Session()
    session.headers.update(headers)
    session.max_redirects = 5 # Limit redirects to avoid loops

    for link in unique_links:
        status = None
        error_type = None
        try:
            # Try HEAD request first (faster)
            response = session.head(link, timeout=10, allow_redirects=True)
            if not response.ok: # Check status code >= 400
                # Fallback to GET if HEAD fails or is disallowed (e.g., 405 Method Not Allowed)
                if response.status_code in [405, 403, 501] or str(response.status_code).startswith('4'):
                     try:
                         response_get = session.get(link, timeout=12, stream=True) # Slightly longer timeout for GET
                         if not response_get.ok:
                             status = f"GET Status: {response_get.status_code}"
                         response_get.close() # Ensure connection is closed
                     except requests.exceptions.RequestException as e_get:
                          error_type = f"GET Error: {type(e_get).__name__}"
                else: # Keep the original HEAD status if it wasn't a 4xx or method issue
                     status = f"HEAD Status: {response.status_code}"

        except requests.exceptions.Timeout:
            error_type = "Timeout"
        except requests.exceptions.TooManyRedirects:
            error_type = "Too Many Redirects"
        except requests.exceptions.ConnectionError:
             error_type = "Connection Error"
        except requests.exceptions.RequestException as e:
            error_type = f"Error: {type(e).__name__}"
        except Exception as e: # Catch any other unexpected errors
             error_type = f"Unexpected Error: {type(e).__name__}"

        if status or error_type:
             broken.append((link, status or error_type))

        time.sleep(0.05) # Slightly reduce delay

    session.close() # Close the session
    return broken


# --- API Call Functions (Placeholder for Competitor Analysis) ---
@st.cache_data(ttl=3600)
def get_serp_competitors_placeholder(keywords, target_domain, country='us', lang='en'):
    """
    Placeholder function for fetching competitor data.
    Replace this with actual SERP API calls if you have credentials.
    Requires SERP_API_KEY and SERP_API_ENDPOINT secrets/env vars.
    """
    # Display warning only if API keys are not set
    if not SERP_API_KEY or not SERP_API_ENDPOINT:
        st.warning(f"""
        **Competitor Identification Disabled:**
        This requires a paid SERP API (e.g., SerpApi, ValueSERP, ScraperAPI) integration.
        Using placeholder data for demonstration. Set `SERP_API_KEY` and `SERP_API_ENDPOINT`
        secrets in Streamlit Cloud settings and update this function to enable real competitor fetching.
        """)
        # Return empty data if keys are missing to avoid confusion
        return [], {}, "SERP API credentials not configured."


    # Simulate finding competitors for the first keyword
    primary_keyword = keywords[0] if keywords else "seo"
    st.write(f"(Simulating competitor search for: '{primary_keyword}')")

    # --- Example using a hypothetical SERP API call ---
    # params = {
    #     'api_key': SERP_API_KEY,
    #     'q': primary_keyword,
    #     'location': country, # Adjust param names based on API provider
    #     'hl': lang,
    #     'num': 10 # Get top 10 results
    # }
    # try:
    #     response = requests.get(SERP_API_ENDPOINT, params=params, timeout=20)
    #     response.raise_for_status()
    #     serp_data = response.json()
    #
    #     # --- PARSE REAL API RESPONSE ---
    #     # This part depends heavily on the specific SERP API's output format
    #     organic_results = serp_data.get('organic_results', [])
    #     competitors = [
    #         {
    #             'position': r.get('position'),
    #             'title': r.get('title'),
    #             'link': r.get('link'),
    #             'domain': get_domain(r.get('link'))
    #          }
    #         for r in organic_results
    #         if r.get('link') and get_domain(r.get('link')) != target_domain # Exclude target domain
    #     ][:5] # Take top 5 competitors
    #
    #     serp_features = { # Extract other features based on API output
    #          'related_questions': serp_data.get('related_questions', []),
    #          'featured_snippet': serp_data.get('featured_snippet', None),
    #          # ... other features ...
    #     }
    #     return competitors, serp_features, None
    #
    # except requests.exceptions.RequestException as e:
    #      st.error(f"SERP API request failed: {e}")
    #      return [], {}, f"SERP API request failed: {e}"
    # except Exception as e:
    #      st.error(f"Error processing SERP API response: {e}")
    #      return [], {}, f"Error processing SERP API response: {e}"
    # --- End of Hypothetical API Call ---


    # --- Using Placeholder data if API call above is commented out ---
    competitors_list_sim = [
        {'position': 1, 'title': f'Ultimate Guide to {primary_keyword} - Competitor A', 'link': 'https://competitor-a.com/ultimate-guide', 'domain': 'competitor-a.com'},
        {'position': 2, 'title': f'{primary_keyword} Strategies for 2025 - Competitor B', 'link': 'https://competitor-b.com/strategies', 'domain': 'competitor-b.com'},
        {'position': 3, 'title': f'Why {primary_keyword} Matters - Competitor C Blog', 'link': 'https://blog.competitor-c.com/why-seo', 'domain': 'blog.competitor-c.com'},
    ]
    serp_features_sim = {
        'search_information': {'query_displayed': primary_keyword},
        'organic_results': [c for c in competitors_list_sim if c['domain'] != target_domain][:3],
        'related_questions': [
            {'question': f'What is {primary_keyword}?'}, {'question': f'How to improve {primary_keyword}?'}, {'question': f'Best tools for {primary_keyword}?'}
        ] if 'seo' in primary_keyword else [],
        'featured_snippet': None, 'knowledge_graph': None, 'inline_videos': None, 'top_stories': None, 'local_pack': None,
    }
    actual_competitors_sim = [c for c in competitors_list_sim if c['domain'] != target_domain]
    return actual_competitors_sim[:3], serp_features_sim, None # Return top 3 simulated competitors


# --- Analysis Functions ---

def analyze_on_page(soup, url, keywords, html_content):
    """Analyzes On-Page SEO elements."""
    results = {}
    recommendations = []
    base_domain = get_domain(url)

    # 1. Title Tag
    try:
        title_tag = soup.find('title')
        title_text = title_tag.string.strip() if title_tag and title_tag.string else ""
        results['title_presence'] = {'value': bool(title_text), 'pass': bool(title_text), 'details': title_text}
        results['title_length'] = {'value': len(title_text), 'pass': 0 < len(title_text) <= TITLE_MAX_LEN, 'details': f"Length: {len(title_text)} chars (Recommended: <={TITLE_MAX_LEN})"}
        title_kw_found = [kw for kw in keywords if kw.lower() in title_text.lower()] if title_text else []
        results['title_keywords'] = {'value': bool(title_kw_found), 'pass': bool(title_kw_found), 'details': f"Found: {', '.join(title_kw_found) if title_kw_found else 'None'}"}
        modifiers = ['guide', 'best', 'review', 'how to', 'checklist', 'tutorial', str(pd.Timestamp.now().year)]
        title_modifier_found = any(mod in title_text.lower() for mod in modifiers) if title_text else False
        results['title_modifiers'] = {'value': title_modifier_found, 'pass': title_modifier_found, 'details': "Check for words like 'guide', 'best', year, etc."}
        if not bool(title_text): recommendations.append("CRITICAL: Add a compelling Title Tag.")
        elif not results['title_length']['pass']: recommendations.append("Improve Title Tag: Adjust length to be under 60 characters.")
        elif not results['title_keywords']['pass']: recommendations.append("Improve Title Tag: Include primary target keywords near the beginning.")
    except Exception as e:
        st.error(f"Error analyzing Title tag: {e}")
        results['title_presence'] = results['title_length'] = results['title_keywords'] = results['title_modifiers'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    # 2. Meta Description
    try:
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        meta_desc_content = meta_desc_tag.get('content', '').strip() if meta_desc_tag else ""
        results['meta_desc_presence'] = {'value': bool(meta_desc_content), 'pass': bool(meta_desc_content), 'details': meta_desc_content[:100] + "..." if meta_desc_content else "Missing"}
        results['meta_desc_length'] = {'value': len(meta_desc_content), 'pass': 0 < len(meta_desc_content) <= META_DESC_MAX_LEN, 'details': f"Length: {len(meta_desc_content)} chars (Recommended: ~155-{META_DESC_MAX_LEN})"}
        meta_kw_found = [kw for kw in keywords if kw.lower() in meta_desc_content.lower()] if meta_desc_content else []
        results['meta_desc_keywords'] = {'value': bool(meta_kw_found), 'pass': bool(meta_kw_found), 'details': f"Found: {', '.join(meta_kw_found) if meta_kw_found else 'None'}"}
        if not bool(meta_desc_content): recommendations.append("CRITICAL: Add a unique and informative Meta Description.")
        elif not results['meta_desc_length']['pass']: recommendations.append("Improve Meta Description: Adjust length to be ~155-160 characters.")
        elif not results['meta_desc_keywords']['pass']: recommendations.append("Improve Meta Description: Include target keywords naturally.")
    except Exception as e:
         st.error(f"Error analyzing Meta Description: {e}")
         results['meta_desc_presence'] = results['meta_desc_length'] = results['meta_desc_keywords'] = {'value': 'Error', 'pass': False, 'details': str(e)}


    # 3. Header Tags (H1-H6)
    try:
        headers = {'H1': [], 'H2': [], 'H3': [], 'H4': [], 'H5': [], 'H6': []}
        for i in range(1, 7):
            for header in soup.find_all(f'h{i}'):
                headers[f'H{i}'].append(header.get_text(strip=True))
        h1_tags = headers['H1']
        results['h1_presence'] = {'value': len(h1_tags) == 1, 'pass': len(h1_tags) == 1, 'details': f"Found {len(h1_tags)} H1 tag(s). Recommended: Exactly 1."}
        h1_text = h1_tags[0] if len(h1_tags) >= 1 else "" # Take first H1 if multiple exist for keyword check
        h1_kw_found = [kw for kw in keywords if kw.lower() in h1_text.lower()] if h1_text else []
        results['h1_keywords'] = {'value': bool(h1_kw_found), 'pass': bool(h1_kw_found), 'details': f"Keywords in H1: {', '.join(h1_kw_found) if h1_kw_found else 'None'}"}
        results['h1_content'] = {'value': h1_text, 'pass': bool(h1_text), 'details': h1_text[:100] + "..." if h1_text else "N/A"}
        if len(h1_tags) != 1: recommendations.append("CRITICAL: Ensure there is exactly one H1 tag on the page.")
        elif not h1_kw_found: recommendations.append("Improve H1 Tag: Include primary target keywords.")
        results['h2_h6_structure'] = {'value': any(len(headers[f'H{i}']) > 0 for i in range(2, 7)), 'pass': any(len(headers[f'H{i}']) > 0 for i in range(2, 7)), 'details': f"Found H2s: {len(headers['H2'])}, H3s: {len(headers['H3'])}, etc. Used for structure?"}
        if not results['h2_h6_structure']['pass']: recommendations.append("Improve Content Structure: Use H2-H6 tags to organize content logically.")
    except Exception as e:
        st.error(f"Error analyzing Header tags: {e}")
        results['h1_presence'] = results['h1_keywords'] = results['h1_content'] = results['h2_h6_structure'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    # 4. Content Body & Readability
    main_content_text = ""
    word_count = 0
    try:
        # Use readability-lxml for better main content extraction
        doc = Document(html_content)
        summary_html = doc.summary()
        # readable_title = doc.title() # Can be useful but not strictly needed here
        summary_soup = BeautifulSoup(summary_html, 'lxml')
        main_content_text = summary_soup.get_text(separator=' ', strip=True)
        word_count = len(main_content_text.split())
    except Exception as e:
        st.warning(f"Readability processing failed: {e}. Analysis might be less accurate, falling back to full body text.")
        # Fallback to trying full body text if readability fails
        try:
             main_content_text = soup.get_text(separator=' ', strip=True)
             word_count = len(main_content_text.split())
        except Exception as e_fallback:
             st.error(f"Could not extract text content: {e_fallback}")
             main_content_text = ""
             word_count = 0
             results['content_word_count'] = {'value': 'Error', 'pass': False, 'details': str(e_fallback)}
             results['keyword_density'] = {'value': 'Error', 'pass': False, 'details': 'N/A'}
             results['keyword_prominence'] = {'value': 'Error', 'pass': False, 'details': 'N/A'}


    if word_count > 0: # Proceed only if text was extracted
        results['content_word_count'] = {'value': word_count, 'pass': word_count > 300, 'details': f"{word_count} words (approx. main content). Compare vs competitors."}
        keyword_counts = {kw: main_content_text.lower().count(kw.lower()) for kw in keywords}
        total_kw_count = sum(keyword_counts.values())
        density = (total_kw_count / word_count * 100) if word_count > 0 else 0
        results['keyword_density'] = {'value': f"{density:.2f}%", 'pass': 0.5 <= density <= 2.5, 'details': f"Found keywords: {keyword_counts}. Density approx {density:.2f}% (Guideline: 0.5-2.5%, focus on natural use)."}
        first_100_words = " ".join(main_content_text.split()[:100])
        kw_in_first_100 = [kw for kw in keywords if kw.lower() in first_100_words.lower()]
        results['keyword_prominence'] = {'value': bool(kw_in_first_100), 'pass': bool(kw_in_first_100), 'details': f"Keywords in first 100 words: {', '.join(kw_in_first_100) if kw_in_first_100 else 'None'}."}
        if word_count < 500: recommendations.append("Improve Content Depth: Content seems thin (<500 words). Consider expanding if appropriate.")
        if not results['keyword_prominence']['pass']: recommendations.append("Improve Keyword Placement: Try to include a primary keyword naturally within the first 100 words.")

        # Simplified Readability Metrics
        try:
            sentences = len(re.findall(r'[.!?]+', main_content_text)) # Approx sentence count
            words = word_count
            avg_sentence_length = words / sentences if sentences > 0 else words
            paragraphs = main_content_text.count('\n\n') + 1 # Estimate paragraphs
            avg_paragraph_length = words / paragraphs if paragraphs > 0 else words

            results['readability'] = {
                'value': f"~{avg_sentence_length:.1f} words/sentence, ~{avg_paragraph_length:.0f} words/paragraph",
                'pass': avg_sentence_length < 25 and avg_paragraph_length < 150, # Example thresholds
                'details': "Lower is often better. Check for short sentences/paragraphs, lists, formatting. (Install 'textstat' for Flesch score)."
            }
            if not results['readability']['pass']: recommendations.append("Improve Readability: Break down long sentences/paragraphs. Use lists/formatting.")
        except Exception as e:
            results['readability'] = {'value': 'Error', 'pass': False, 'details': f"Could not calculate readability: {e}"}
            recommendations.append("Warning: Could not calculate readability metrics.")
    else: # Handle case where no text could be extracted
         if 'content_word_count' not in results: # Ensure keys exist even on failure
             results['content_word_count'] = {'value': 0, 'pass': False, 'details': 'Could not extract content.'}
             results['keyword_density'] = {'value': 'N/A', 'pass': False, 'details': 'N/A'}
             results['keyword_prominence'] = {'value': 'N/A', 'pass': False, 'details': 'N/A'}
             results['readability'] = {'value': 'N/A', 'pass': False, 'details': 'N/A'}
         recommendations.append("Error: Could not extract page content for analysis.")


    # 5. Image SEO
    try:
        images = soup.find_all('img')
        results['image_count'] = {'value': len(images), 'pass': True, 'details': f"Found {len(images)} images."}
        if len(images) > 0:
            alt_missing = 0
            alt_generic = 0
            descriptive_filenames = 0
            generic_filenames = ['image', 'pic', 'logo', 'icon', 'banner', 'photo', 'img', 'bg', 'background', 'thumb', 'sprite', 'spacer']
            for img in images:
                alt = img.get('alt', '').strip()
                if not alt:
                    alt_missing += 1
                elif len(alt) < 5 or alt.lower() in ['image', 'picture', 'logo', 'icon', 'graphic', 'photo', 'alt', 'spacer', 'banner', 'button']:
                    alt_generic += 1
                src = img.get('src', '')
                filename = src.split('/')[-1].split('?')[0].split('.')[0].lower() if src else "" # Clean filename
                if filename and not any(gf in filename for gf in generic_filenames) and len(filename) > 3: # Basic check for descriptive name
                    descriptive_filenames += 1

            alt_text_coverage = (len(images) - alt_missing) / len(images) * 100
            results['image_alt_text'] = {'value': f"{alt_text_coverage:.0f}%", 'pass': alt_text_coverage >= 90, 'details': f"{len(images) - alt_missing}/{len(images)} images have alt text. Missing: {alt_missing}, Potentially Generic: {alt_generic}."}
            results['image_filenames'] = {'value': f"{descriptive_filenames}/{len(images)}", 'pass': descriptive_filenames / len(images) >= 0.75, 'details': "Check if filenames are descriptive (e.g., 'black-cat.jpg' vs 'img_123.jpg')."}
            if alt_text_coverage < 90: recommendations.append("Improve Image SEO: Add descriptive alt text to all important images.")
            if descriptive_filenames / len(images) < 0.75: recommendations.append("Improve Image SEO: Use descriptive file names for images.")
        else:
             results['image_alt_text'] = {'value': 'N/A', 'pass': True, 'details': "No images found."}
             results['image_filenames'] = {'value': 'N/A', 'pass': True, 'details': "No images found."}
        recommendations.append("Optimize Images: Ensure images are compressed and use modern formats (e.g., WebP) for faster loading (Manual Check Required).")
    except Exception as e:
         st.error(f"Error analyzing Images: {e}")
         results['image_count'] = results['image_alt_text'] = results['image_filenames'] = {'value': 'Error', 'pass': False, 'details': str(e)}


    # 6. Links
    internal_links = []
    external_links = []
    all_links_href = []
    try:
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:') or href.startswith('tel:'):
                 continue # Skip invalid, anchor, JS, mailto, tel links

            try:
                abs_url = urljoin(url, href) # Resolve relative URLs
                parsed_abs = urlparse(abs_url)
                # Ensure it's a valid, absolute HTTP/HTTPS URL
                if parsed_abs.scheme in ['http', 'https'] and parsed_abs.netloc:
                     all_links_href.append(abs_url)
                     link_domain = parsed_abs.netloc
                     anchor_text = link.get_text(strip=True)
                     if link_domain == base_domain:
                         internal_links.append({'url': abs_url, 'anchor': anchor_text})
                     elif link_domain and link_domain != base_domain:
                         external_links.append({'url': abs_url, 'anchor': anchor_text})
            except Exception:
                 continue # Ignore errors resolving/parsing specific URLs

        results['internal_links_count'] = {'value': len(internal_links), 'pass': len(internal_links) > 0 if word_count > 300 else True, 'details': f"Found {len(internal_links)} internal links."} # Pass if content is very short
        results['external_links_count'] = {'value': len(external_links), 'pass': True, 'details': f"Found {len(external_links)} external links."}
        internal_anchors = [l['anchor'][:50].replace('\n', ' ').strip() + ('...' if len(l['anchor']) > 50 else '') for l in internal_links if l.get('anchor')]
        external_anchors = [l['anchor'][:50].replace('\n', ' ').strip() + ('...' if len(l['anchor']) > 50 else '') for l in external_links if l.get('anchor')]
        results['internal_anchors'] = {'value': ', '.join(list(set(internal_anchors))[:5]), 'pass': True, 'details': "Sample internal link anchors. Check for relevance and diversity."}
        results['external_anchors'] = {'value': ', '.join(list(set(external_anchors))[:5]), 'pass': True, 'details': "Sample external link anchors. Check relevance/authority."}

        if len(internal_links) == 0 and word_count > 500: recommendations.append("Improve Internal Linking: Add relevant internal links.")
        # Only recommend external links if content is substantial and none exist
        # if len(external_links) == 0 and word_count > 1000: recommendations.append("Consider External Linking: If appropriate, link out to relevant external resources.")

        # Broken Links (checked separately)
        results['broken_links'] = {'value': 'Pending check...', 'pass': True, 'details': 'Checking links...' } # Placeholder

    except Exception as e:
        st.error(f"Error analyzing Links: {e}")
        results['internal_links_count'] = results['external_links_count'] = results['internal_anchors'] = results['external_anchors'] = results['broken_links'] = {'value': 'Error', 'pass': False, 'details': str(e)}


    return results, recommendations, all_links_href

def analyze_technical(soup, url, html_content, ttfb, keywords): # Pass keywords
    """Analyzes Technical SEO aspects (excluding PageSpeed Insights)."""
    results = {}
    recommendations = []
    parsed_url = urlparse(url)

    # 1. Page Load Speed (Basic TTFB Only)
    try:
        results['ttfb'] = {'value': f"{ttfb:.3f}s" if ttfb is not None else "N/A", 'pass': ttfb < 0.8 if ttfb is not None else None, 'details': f"Time To First Byte (TTFB): {ttfb:.3f}s (Target <0.8s. Not a full speed measure!)."}
        if ttfb is not None and ttfb >= 0.8: recommendations.append("Improve Server Response Time: TTFB is high (>0.8s). Investigate server/hosting.")
        # Always recommend manual check
        recommendations.append("Review Page Speed Manually: Use external tools (Google PageSpeed web UI, WebPageTest.org) for detailed speed/CWV analysis.")
    except Exception as e:
        st.error(f"Error processing TTFB: {e}")
        results['ttfb'] = {'value': 'Error', 'pass': False, 'details': str(e)}


    # 2. Mobile-Friendliness (Basic Checks)
    try:
        viewport_tag = soup.find('meta', attrs={'name': 'viewport'})
        results['mobile_viewport'] = {'value': bool(viewport_tag), 'pass': bool(viewport_tag), 'details': f"Viewport meta tag found: {'Yes' if viewport_tag else 'No'}"}
        if not viewport_tag: recommendations.append("Improve Mobile Experience: Add a viewport meta tag (e.g., <meta name='viewport' content='width=device-width, initial-scale=1'>).")
        # Basic heuristic checks
        results['mobile_fonts'] = {'value': 'Manual Check', 'pass': True, 'details': f"Manually check CSS/DevTools for font sizes >= {MIN_FONT_SIZE_PX}px."}
        has_potential_interstitial = 'popup' in html_content.lower() or 'modal' in html_content.lower() or 'interstitial' in html_content.lower()
        results['mobile_interstitials'] = {'value': 'Basic Check', 'pass': not has_potential_interstitial, 'details': f"Potentially intrusive elements? {'Yes' if has_potential_interstitial else 'No'} (Manual check recommended)."}
        if has_potential_interstitial: recommendations.append("Review Mobile Experience: Ensure popups/interstitials are not intrusive.")
        results['mobile_tap_targets'] = {'value': 'Manual Check', 'pass': True, 'details': f"Manually check/DevTools: Ensure interactive elements are large enough (e.g., >= 48x48px)."}
    except Exception as e:
         st.error(f"Error analyzing Mobile Friendliness basics: {e}")
         results['mobile_viewport'] = results['mobile_fonts'] = results['mobile_interstitials'] = results['mobile_tap_targets'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    # 3. URL Structure
    try:
        url_str = str(url)
        results['url_length'] = {'value': len(url_str), 'pass': len(url_str) < 100, 'details': f"URL Length: {len(url_str)} chars (Shorter is better)."}
        # Use keywords passed to the function
        url_kw_found = [kw for kw in keywords if kw.lower() in url_str.lower()]
        results['url_keywords'] = {'value': bool(url_kw_found), 'pass': bool(url_kw_found), 'details': f"Keywords in URL: {', '.join(url_kw_found) if url_kw_found else 'None'}."}
        path_part = parsed_url.path
        simple_path = not re.search(r'[_?&=%]', path_part) # No underscores or common params
        if simple_path and not re.search(r'-', path_part) and len(path_part.split('/')) > 2 and len(path_part.split('/')[-1]) > 15:
             simple_path = False # Heuristic: Long path segment without hyphens? Less readable.
        results['url_readability'] = {'value': 'Readable' if simple_path else 'Less Readable', 'pass': simple_path, 'details': "Uses hyphens? Avoids underscores/parameters? Clear words?"}
        if len(url_str) >= 100: recommendations.append("Consider URL Structure: URL is long (>100 chars). Shorter URLs preferred.")
        if not simple_path: recommendations.append("Improve URL Structure: Use hyphens. Avoid underscores, excessive parameters, long segments.")
    except Exception as e:
        st.error(f"Error analyzing URL Structure: {e}")
        results['url_length'] = results['url_keywords'] = results['url_readability'] = {'value': 'Error', 'pass': False, 'details': str(e)}

    # 4. HTTPS Security
    try:
        is_https = parsed_url.scheme == 'https'
        results['https'] = {'value': 'Yes' if is_https else 'No', 'pass': is_https, 'details': f"URL uses HTTPS: {'Yes' if is_https else 'No - CRITICAL'}"}
        if not is_https: recommendations.append("CRITICAL SECURITY RISK: Migrate site to HTTPS immediately.")
    except Exception as e:
         st.error(f"Error checking HTTPS: {e}")
         results['https'] = {'value': 'Error', 'pass': False, 'details': str(e)}


    # 5. Schema Markup / Structured Data
    try:
        schema_found = []
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                # Handle potential comments within script tags if needed
                script_content = script.string
                if script_content:
                    data = json.loads(script_content)
                    if isinstance(data, list):
                        for item in data:
                            schema_type = item.get('@type') if isinstance(item, dict) else None
                            if schema_type: schema_found.append(f"JSON-LD: {schema_type}")
                    elif isinstance(data, dict):
                        schema_type = data.get('@type')
                        if schema_type: schema_found.append(f"JSON-LD: {schema_type}")
            except Exception: schema_found.append("JSON-LD: Error parsing/reading")
        microdata_items = soup.find_all(itemscope=True)
        for item in microdata_items:
            item_type = item.get('itemtype')
            if item_type:
                schema_type = item_type.split('/')[-1]
                schema_found.append(f"Microdata: {schema_type}")
        results['schema_markup'] = {'value': ', '.join(sorted(list(set(schema_found)))) if schema_found else 'None Found', 'pass': bool(schema_found), 'details': "Presence of Schema.org markup."}
        if not schema_found: recommendations.append("Implement Schema Markup: Add relevant structured data (Article, FAQ, Product, etc.).")
    except Exception as e:
        st.error(f"Error analyzing Schema Markup: {e}")
        results['schema_markup'] = {'value': 'Error', 'pass': False, 'details': str(e)}


    # 6. Crawlability/Indexability
    try:
        # Robots.txt Check (Placeholder to be filled after fetch)
        results['robots_txt_check'] = {'value': 'Pending check...', 'pass': True, 'details': 'Checking robots.txt...'}

        # Meta Robots Tag
        meta_robots = soup.find('meta', attrs={'name': 'robots'})
        robots_content = meta_robots['content'].lower() if meta_robots and 'content' in meta_robots.attrs else ''
        noindex = 'noindex' in robots_content
        nofollow = 'nofollow' in robots_content # Less critical for score but good to know
        results['meta_robots'] = {'value': robots_content if robots_content else 'Not specified (Implies index, follow)', 'pass': not noindex, 'details': f"Meta Robots: {'noindex' if noindex else 'index'}, {'nofollow' if nofollow else 'follow'}"}
        if noindex: recommendations.append("CRITICAL: Page has 'noindex' tag, preventing indexing. Remove if unintentional.")

        # Canonical Tag
        canonical_tag = soup.find('link', rel='canonical')
        canonical_url = canonical_tag.get('href', '').strip() if canonical_tag else ''
        results['canonical_tag'] = {'value': canonical_url if canonical_url else 'Not specified', 'pass': True, 'details': f"Canonical URL: {canonical_url if canonical_url else 'None'}"}
        # Check if canonical URL is substantially different (ignore http/https, trailing slash differences)
        norm_url = url.replace('https://', 'http://').rstrip('/')
        norm_canon = canonical_url.replace('https://', 'http://').rstrip('/')
        if canonical_url and norm_canon != norm_url:
            recommendations.append(f"Review Canonical Tag: Canonical URL ({canonical_url}) differs from analyzed URL. Ensure intentional.")
        elif not canonical_url:
            recommendations.append(f"Add Canonical Tag: Specify a self-referencing canonical tag.")
    except Exception as e:
        st.error(f"Error analyzing Crawlability/Indexability tags: {e}")
        results['robots_txt_check'] = results['meta_robots'] = results['canonical_tag'] = {'value': 'Error', 'pass': False, 'details': str(e)}


    return results, recommendations


def calculate_overall_score(on_page_results, tech_results):
    """ Calculates a weighted score based on analysis results (No PageSpeed). """
    # Define weights (adjust as needed, total = 1.0 for each category)
    weights = {
        'on_page': 0.6, 'technical': 0.4,
        # On-page sub-weights (ensure sum ~1.0)
        'title': 0.15, 'meta_desc': 0.10, 'h1': 0.10, 'content': 0.15,
        'keywords':0.10, 'readability': 0.08, 'images': 0.12, 'links': 0.15,
        'broken_links': 0.05, # Added broken links check
        # Technical sub-weights (ensure sum ~1.0)
        'https': 0.30, 'crawl': 0.30, 'url': 0.15, 'schema': 0.10, 'mobile': 0.10,
        'ttfb': 0.05 # Added basic TTFB check
    }

    scores = {'on_page': 0, 'technical': 0}

    def get_pass_score(result_dict, key):
        # Helper to safely get pass status (returns 0 if key missing or pass is False/None)
        return 1 if result_dict.get(key, {}).get('pass', False) else 0

    # --- On-Page Scoring ---
    op_score = 0
    op_max = 0
    op_max += weights['title'] * 3; op_score += weights['title'] * (get_pass_score(on_page_results, 'title_presence') + get_pass_score(on_page_results, 'title_length') + get_pass_score(on_page_results, 'title_keywords'))
    op_max += weights['meta_desc'] * 3; op_score += weights['meta_desc'] * (get_pass_score(on_page_results, 'meta_desc_presence') + get_pass_score(on_page_results, 'meta_desc_length') + get_pass_score(on_page_results, 'meta_desc_keywords'))
    op_max += weights['h1'] * 2; op_score += weights['h1'] * (get_pass_score(on_page_results, 'h1_presence') + get_pass_score(on_page_results, 'h1_keywords'))
    op_max += weights['keywords'] * 2; op_score += weights['keywords'] * (get_pass_score(on_page_results, 'keyword_density') + get_pass_score(on_page_results, 'keyword_prominence'))
    op_max += weights['content'] * 1; op_score += weights['content'] * get_pass_score(on_page_results, 'content_word_count')
    op_max += weights['readability'] * 1; op_score += weights['readability'] * get_pass_score(on_page_results, 'readability')
    op_max += weights['images'] * 2; op_score += weights['images'] * (get_pass_score(on_page_results, 'image_alt_text') + get_pass_score(on_page_results, 'image_filenames'))
    op_max += weights['links'] * 1; op_score += weights['links'] * get_pass_score(on_page_results, 'internal_links_count')
    op_max += weights['broken_links'] * 1; op_score += weights['broken_links'] * get_pass_score(on_page_results, 'broken_links')

    scores['on_page'] = (op_score / op_max * 100) if op_max > 0 else 0

    # --- Technical Scoring ---
    tech_score = 0
    tech_max = 0
    tech_max += weights['https'] * 1; tech_score += weights['https'] * get_pass_score(tech_results, 'https')
    robots_allowed = get_pass_score(tech_results, 'robots_txt_check') # Pass = Allowed
    tech_max += weights['crawl'] * 2; tech_score += weights['crawl'] * (get_pass_score(tech_results, 'meta_robots') + robots_allowed)
    tech_max += weights['url'] * 2; tech_score += weights['url'] * (get_pass_score(tech_results, 'url_length') + get_pass_score(tech_results, 'url_readability'))
    tech_max += weights['schema'] * 1; tech_score += weights['schema'] * get_pass_score(tech_results, 'schema_markup')
    # Basic mobile checks combined
    tech_max += weights['mobile'] * 2; tech_score += weights['mobile'] * (get_pass_score(tech_results, 'mobile_viewport') + get_pass_score(tech_results, 'mobile_interstitials'))
    tech_max += weights['ttfb'] * 1; tech_score += weights['ttfb'] * get_pass_score(tech_results, 'ttfb')

    scores['technical'] = (tech_score / tech_max * 100) if tech_max > 0 else 0

    # --- Overall Score ---
    # Ensure scores are within 0-100 range before weighting
    scores['on_page'] = max(0, min(100, scores['on_page']))
    scores['technical'] = max(0, min(100, scores['technical']))

    overall_score = (scores['on_page'] * weights['on_page']) + (scores['technical'] * weights['technical'])
    scores['overall'] = int(round(overall_score)) # Round to nearest integer

    return scores


def generate_recommendations(on_page_recs, tech_recs, on_page_results, tech_results):
    """ Combines and prioritizes recommendations (No PageSpeed). """
    prioritized = []
    processed_recs = set() # Keep track of recommendations already added

    def add_rec(rec, priority_level):
        rec_text = rec.split(': ')[-1] # Get the core message
        if rec_text not in processed_recs:
            prioritized.append(f"{priority_level}: {rec}")
            processed_recs.add(rec_text)

    # Critical Technical First
    if not tech_results.get('https', {}).get('pass'): add_rec(tech_recs.pop(tech_recs.index(next(r for r in tech_recs if 'HTTPS' in r))), "🔴 CRITICAL")
    if not tech_results.get('meta_robots', {}).get('pass'): add_rec(tech_recs.pop(tech_recs.index(next(r for r in tech_recs if 'noindex' in r))), "🔴 CRITICAL")
    if not tech_results.get('robots_txt_check', {}).get('pass'): add_rec(tech_recs.pop(tech_recs.index(next(r for r in tech_recs if 'robots.txt' in r))), "🔴 CRITICAL")

    # Critical On-Page
    if not on_page_results.get('title_presence', {}).get('pass'): add_rec(on_page_recs.pop(on_page_recs.index(next(r for r in on_page_recs if 'Title Tag' in r and 'Add' in r))), "🟠 HIGH")
    if not on_page_results.get('meta_desc_presence', {}).get('pass'): add_rec(on_page_recs.pop(on_page_recs.index(next(r for r in on_page_recs if 'Meta Description' in r and 'Add' in r))), "🟠 HIGH")
    if not on_page_results.get('h1_presence', {}).get('pass'): add_rec(on_page_recs.pop(on_page_recs.index(next(r for r in on_page_recs if 'H1 tag' in r))), "🟠 HIGH")

    # Broken Links
    if not on_page_results.get('broken_links', {}).get('pass'): add_rec(on_page_recs.pop(on_page_recs.index(next(r for r in on_page_recs if 'Broken Links' in r))), "🟠 HIGH")


    # Combine remaining, assign priorities based on keywords
    remaining_recs = tech_recs + on_page_recs
    for rec in remaining_recs:
        prefix = "⚪️ LOW" # Default low
        rec_lower = rec.lower()
        # Medium priority examples
        if any(kw in rec_lower for kw in ["improve readability", "improve url", "server response time", "review canonical", "internal linking", "alt text", "image filenames"]):
             prefix = "🔵 MEDIUM"
        # High priority examples (already covered critical/presence, focus on missing keywords etc)
        if any(kw in rec_lower for kw in ["improve title tag", "improve meta description", "improve h1 tag"]):
            prefix = "🟠 HIGH"
        if any(kw in rec_lower for kw in ["add canonical tag", "add schema markup", "mobile experience"]):
             prefix = "🔵 MEDIUM" # Usually improvements, not critical failures

        add_rec(rec, prefix)


    # Ensure manual speed check is always mentioned if not already covered
    if "Review Page Speed Manually" not in processed_recs:
         add_rec("Review Page Speed Manually: Use external tools (Google PageSpeed web UI, WebPageTest.org) for detailed speed/CWV analysis.", "🔵 MEDIUM")


    # Sort by priority symbol then alphabetically
    prioritized.sort(key=lambda x: (x.split(':')[0], x.split(': ')[1]))


    return prioritized[:15] # Limit recommendations


# --- Visualization Functions ---

def create_score_gauge(score, title):
    """Creates a simple gauge-like visualization using Altair."""
    if score is None: score = 0
    score = max(0, min(100, int(score)))
    color_scale = alt.Scale(domain=[0, 50, 90, 100], range=['#F44336', '#FFC107', '#4CAF50', '#4CAF50']) # Red, Yellow, Green
    base = alt.Chart(pd.DataFrame({'value': [100]})).mark_arc(outerRadius=100, innerRadius=80, endAngle=3.14).encode(
        theta=alt.Theta("value", stack=True, scale=alt.Scale(range=[0, 3.14])), color=alt.value('#ddd'), order=alt.Order("value", sort="descending")
    )
    score_arc = alt.Chart(pd.DataFrame({'value': [score]})).mark_arc(outerRadius=100, innerRadius=80).encode(
        theta=alt.Theta("value", scale=alt.Scale(range=[0, 3.14])), color=alt.Color("value", scale=color_scale, legend=None), order=alt.Order("value", sort="descending")
    )
    text = alt.Chart(pd.DataFrame({'value': [score]})).mark_text(dy=0, fontSize=30, fontWeight='bold').encode(
        text=alt.Text("value", format=".0f"), color=alt.Color("value", scale=color_scale, legend=None)
    )
    chart = alt.layer(base, score_arc, text).properties(title=title).configure_view(strokeWidth=0)
    return chart

def create_comparison_bar_chart(df_compare, metric, title):
    """ Creates a grouped bar chart for comparing target vs competitors."""
    if metric not in df_compare.columns:
         st.caption(f"Metric '{metric}' not available for comparison chart.")
         return None

    try:
        # Ensure numeric, handle potential strings like 'Yes'/'No' for Schema
        if df_compare[metric].dtype == 'object':
             if 'Yes' in df_compare[metric].unique() or 'No' in df_compare[metric].unique():
                  # Basic conversion for boolean-like strings if needed for a chart
                  df_compare[metric] = df_compare[metric].replace({'Yes': 1, 'No': 0})
        df_compare[metric] = pd.to_numeric(df_compare[metric], errors='coerce')
        df_compare = df_compare.dropna(subset=[metric])
    except Exception as e:
        st.caption(f"Could not process metric '{metric}' for chart: {e}")
        return None


    if df_compare.empty:
        st.caption(f"Not enough data to display {title} chart.")
        return None

    chart = alt.Chart(df_compare).mark_bar().encode(
        x=alt.X('Competitor', sort='-y', title=None, axis=alt.Axis(labels=False)), # Hide competitor names on X if too many
        y=alt.Y(metric, title=metric),
        color=alt.Color('Competitor', legend=alt.Legend(title="Site")),
        tooltip=['Competitor', metric]
    ).properties(title=title)
    return chart

# --- Main Streamlit App ---

def main():
    st.set_page_config(page_title="Single Page SEO Analyzer", layout="wide")

    st.title("📄 Single Page SEO Analyzer & Competitor Benchmark")
    st.markdown("Analyze on-page and technical SEO factors (**excluding** Google PageSpeed Insights / Core Web Vitals). Compares against competitors (using simulated data).")

    # --- Inputs ---
    st.sidebar.header("Inputs")
    url_input = st.sidebar.text_input("Target Webpage URL", placeholder="https://www.example.com/page")
    keywords_input = st.sidebar.text_area("Target Keywords (one per line)", placeholder="seo checklist\ntechnical seo guide")
    run_analysis = st.sidebar.button("Analyze Page")

    # --- Limitations and API Key Info ---
    with st.expander("Important Notes & Limitations", expanded=False):
        st.info("""
            * **Single Page Focus:** Analyzes ONLY the provided URL.
            * **No Page Speed Analysis:** Does **NOT** include Google PageSpeed Insights (Core Web Vitals). Manual checks recommended. Basic TTFB is shown.
            * **Competitor Data (Simulated/Optional API):** Real competitor data requires a paid SERP API. Using placeholder data unless `SERP_API_KEY`/`SERP_API_ENDPOINT` secrets are set in Streamlit Cloud.
            * **Dynamic Content:** May not fully capture JavaScript-rendered content.
            * **Qualitative Aspects:** Content quality, intent matching require manual review.
            * **Robots.txt:** Basic parsing; complex rules might be misinterpreted.
        """)
        if not SERP_API_KEY or not SERP_API_ENDPOINT:
            st.warning("⚠️ **Note:** Real competitor analysis requires SERP API secrets set in Streamlit Cloud. Using placeholder data.")
        else:
             st.success("✅ SERP API credentials found (using Secrets/Env Vars). Will attempt real competitor fetch.")


    # --- Analysis Execution ---
    if run_analysis and url_input and keywords_input:
        if not is_valid_url(url_input):
            st.error("Invalid URL format. Please enter a full URL (e.g., https://www.example.com)")
        else:
            keywords_list = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]
            if not keywords_list:
                st.error("Please enter at least one target keyword.")
            else:
                st.success(f"Analyzing **{url_input}** for keywords: **{', '.join(keywords_list)}**")
                analysis_placeholder = st.empty() # Placeholder for status updates
                analysis_placeholder.info("🚀 Starting analysis...")

                # --- Initialize Results ---
                on_page_results, tech_results, competitor_data, serp_features = {}, {}, pd.DataFrame(), {}
                on_page_recommendations, tech_recommendations, final_recommendations = [], [], []
                scores = {'overall': 0, 'on_page': 0, 'technical': 0}


                # Fetch Target URL HTML
                analysis_placeholder.info("➡️ Fetching target page HTML...")
                html_content, status_code, ttfb, fetch_error = fetch_html(url_input)

                if fetch_error:
                    analysis_placeholder.error(f"Analysis Stopped: Failed to fetch target URL ({status_code}): {fetch_error}")
                    st.stop() # Stop execution if fetch fails

                try:
                    soup = BeautifulSoup(html_content, 'lxml')
                    target_domain = get_domain(url_input)

                    # Run On-Page Analysis
                    analysis_placeholder.info("➡️ Analyzing On-Page SEO...")
                    on_page_results, on_page_recommendations, all_links_on_page = analyze_on_page(soup, url_input, keywords_list, html_content)

                    # Run Technical Analysis
                    analysis_placeholder.info("➡️ Analyzing Technical SEO...")
                    tech_results, tech_recommendations = analyze_technical(soup, url_input, html_content, ttfb, keywords_list)

                    # Run External Checks (Robots.txt, Broken Links) concurrently? Maybe not easily in Streamlit
                    analysis_placeholder.info("➡️ Checking Robots.txt...")
                    robots_txt_content, robots_error = fetch_robots_txt(url_input)
                    if robots_error:
                        tech_results['robots_txt_check'] = {'value': 'Error', 'pass': False, 'details': robots_error}
                        # Add recommendation only if not critical error message already present
                        if "Assuming allowed" not in robots_error:
                             tech_recommendations.append(f"Warning: Could not verify robots.txt: {robots_error}")
                        else: # If permissive, pass is True
                             tech_results['robots_txt_check'] = {'value': 'Not Found/Allowed', 'pass': True, 'details': robots_error}

                    else:
                        url_path = urlparse(url_input).path or "/"
                        is_blocked = is_disallowed(robots_txt_content, url_path)
                        tech_results['robots_txt_check'] = {'value': 'Blocked' if is_blocked else 'Allowed', 'pass': not is_blocked, 'details': f"Path '{url_path}' appears {'blocked' if is_blocked else 'allowed'} by robots.txt."}
                        if is_blocked: tech_recommendations.append("CRITICAL: Page might be blocked by robots.txt. Verify rules.")

                    analysis_placeholder.info("➡️ Checking for Broken Links (can take time)...")
                    broken_links_result = check_broken_links(all_links_on_page)
                    # Update results based on check
                    if 'broken_links' in on_page_results: # Check if key exists from initial analysis
                        on_page_results['broken_links'] = {'value': f"{len(broken_links_result)} found", 'pass': len(broken_links_result) == 0, 'details': f"Found {len(broken_links_result)} broken links (4xx/5xx/Error)."}
                        if broken_links_result: on_page_recommendations.append(f"Fix Broken Links: Found {len(broken_links_result)} potentially broken links.")


                    # Competitor Analysis (Placeholder or Real if API Keys Set)
                    analysis_placeholder.info("➡️ Analyzing Competitors (Simulated or via SERP API)...")
                    competitors_list, serp_features, serp_error = get_serp_competitors_placeholder(keywords_list, target_domain)
                    if serp_error and "credentials not configured" not in serp_error: # Show error only if keys ARE set but API failed
                         st.error(f"Competitor analysis failed: {serp_error}")

                    competitor_analysis_results = []
                    if competitors_list:
                        # In a real scenario, fetch and analyze each competitor URL here.
                        # For demo, we use placeholder data. Add more metrics if fetched.
                        for comp in competitors_list:
                            # Basic info available from placeholder/SERP API
                            comp_analysis = {
                                'Competitor': comp.get('domain', 'N/A'), 'URL': comp.get('link'),
                                'Title': comp.get('title','N/A')[:TITLE_MAX_LEN],
                                'Title Length': len(comp.get('title','')),
                                # Metrics below would require fetching each competitor page
                                # 'Word Count': 1500 + comp.get('position',1) * 200, # Dummy
                                # 'Images': 8 + comp.get('position',1), # Dummy
                                # 'Internal Links': 10 + comp.get('position',1)*2, # Dummy
                                # 'Schema Markup?': 'Yes' if comp.get('position',1) % 2 == 1 else 'No' # Dummy
                            }
                            # --- TODO: Add actual fetching/analysis of competitors if implementing real SERP API ---
                            # comp_html, comp_status, _, comp_fetch_error = fetch_html(comp.get('link'))
                            # if comp_html and not comp_fetch_error:
                            #    comp_soup = BeautifulSoup(comp_html, 'lxml')
                            #    comp_on_page, _, _ = analyze_on_page(comp_soup, comp.get('link'), keywords_list, comp_html)
                            #    comp_tech, _ = analyze_technical(comp_soup, comp.get('link'), comp_html, None, keywords_list) # No TTFB here
                            #    comp_analysis['Word Count'] = comp_on_page.get('content_word_count',{}).get('value')
                            #    comp_analysis['Images'] = comp_on_page.get('image_count',{}).get('value')
                            #    comp_analysis['Internal Links'] = comp_on_page.get('internal_links_count',{}).get('value')
                            #    comp_analysis['Schema Markup?'] = 'Yes' if comp_tech.get('schema_markup',{}).get('pass') else 'No'
                            # --- End TODO ---
                            competitor_analysis_results.append(comp_analysis)

                        competitor_data = pd.DataFrame(competitor_analysis_results)


                    # Calculate Scores & Final Recommendations
                    analysis_placeholder.info("➡️ Calculating scores and recommendations...")
                    scores = calculate_overall_score(on_page_results, tech_results)
                    final_recommendations = generate_recommendations(on_page_recommendations, tech_recommendations, on_page_results, tech_results)

                    analysis_placeholder.empty() # Clear status message
                    st.success("✅ Analysis Complete!")

                    # --- Display Results ---
                    st.header("📊 SEO Analysis Results")

                    # Overall Score
                    st.subheader("Overall Performance Score")
                    score_col1, score_col2, score_col3 = st.columns(3)
                    with score_col1:
                         st.altair_chart(create_score_gauge(scores.get('overall'), "Overall Score"), use_container_width=True)
                    with score_col2:
                         st.altair_chart(create_score_gauge(scores.get('on_page'), "On-Page Score"), use_container_width=True)
                    with score_col3:
                         st.altair_chart(create_score_gauge(scores.get('technical'), "Technical Score"), use_container_width=True)

                    # Prioritized Recommendations
                    st.subheader("💡 Prioritized Recommendations")
                    if final_recommendations:
                        for i, rec in enumerate(final_recommendations): st.markdown(f"{i+1}. {rec}")
                    else: st.success("✅ No high-priority recommendations found!")

                    # Weaknesses & Opportunities
                    st.subheader("📉 Weaknesses & Opportunities")
                    weaknesses = [rec for rec in final_recommendations if "CRITICAL" in rec or "HIGH" in rec]
                    opportunities = [rec for rec in final_recommendations if "MEDIUM" in rec or "LOW" in rec]
                    opp_col1, opp_col2 = st.columns(2)
                    with opp_col1:
                        st.markdown("**Key Weaknesses:**")
                        if weaknesses:
                             for w in weaknesses: st.markdown(f"- {w.split(': ', 1)[-1]}") # Show message part
                        else: st.write("None identified.")
                    with opp_col2:
                         st.markdown("**Potential Opportunities:**")
                         if opportunities:
                             for o in opportunities: st.markdown(f"- {o.split(': ', 1)[-1]}")
                         # Add content gap / SERP feature opportunities if SERP API was real
                         if serp_features and serp_features.get('related_questions'):
                              st.markdown("**Content Ideas (from 'People Also Ask'):**")
                              for q in serp_features['related_questions'][:3]: # Show top 3 PAA
                                  st.markdown(f"- {q.get('question', 'N/A')}")

                         else: st.write("No specific content opportunities identified.")

                    # Detailed Analysis Tabs
                    st.subheader("🔍 Detailed Analysis")
                    tab1, tab2, tab3 = st.tabs(["On-Page SEO", "Technical SEO", "Competitor Analysis"])

                    with tab1:
                        st.subheader("On-Page SEO Checklist")
                        # Convert results dict to dataframe format
                        on_page_list = []
                        for key, data in on_page_results.items():
                             status = 'ℹ️ Info'
                             if data.get('pass') is True: status = '✅ Pass'
                             elif data.get('pass') is False: status = '❌ Fail' if 'CRITICAL' in data.get('details', '').upper() else '⚠️ Needs Improvement'
                             # Format parameter name nicely
                             param_name = key.replace('_', ' ').title()
                             on_page_list.append({'Parameter': param_name, 'Result': data.get('details', 'N/A'), 'Status': status})

                        on_page_df = pd.DataFrame(on_page_list)
                        st.dataframe(on_page_df, use_container_width=True, hide_index=True)

                        if broken_links_result:
                            st.subheader("Detected Broken Links / Errors")
                            broken_df = pd.DataFrame(broken_links_result, columns=['URL', 'Status/Error'])
                            st.dataframe(broken_df, use_container_width=True, hide_index=True)

                    with tab2:
                        st.subheader("Technical SEO Checklist")
                         # Convert results dict to dataframe format
                        tech_list = []
                        for key, data in tech_results.items():
                             status = 'ℹ️ Info'
                             if data.get('pass') is True: status = '✅ Pass'
                             elif data.get('pass') is False: status = '❌ CRITICAL' if 'CRITICAL' in data.get('details', '').upper() else '⚠️ Needs Improvement'
                             # Format parameter name nicely
                             param_name = key.replace('_', ' ').title()
                             tech_list.append({'Parameter': param_name, 'Result': data.get('details', 'N/A'), 'Status': status})

                        tech_df = pd.DataFrame(tech_list)
                        st.dataframe(tech_df, use_container_width=True, hide_index=True)


                    with tab3:
                        st.subheader("Competitor Benchmark Report")
                        if not competitor_data.empty:
                            st.markdown("**Top Simulated/Identified Competitors:**")
                            for idx, row in competitor_data.iterrows():
                                st.write(f"{idx+1}. {row['Competitor']} ({row['URL']})")

                            st.markdown("**Comparative Analysis (Basic - Requires fetching competitor pages for more detail):**")

                            # Prepare data for comparison chart (using ONLY data available from target page analysis + placeholder competitor data)
                            target_metrics = {
                                'Competitor': f"Target Page", # Shorter name
                                'URL': url_input,
                                'Title Length': on_page_results.get('title_length',{}).get('value'),
                                'Word Count': on_page_results.get('content_word_count',{}).get('value'),
                                'Images': on_page_results.get('image_count',{}).get('value'),
                                'Internal Links': on_page_results.get('internal_links_count',{}).get('value'),
                                'Schema Markup?': 'Yes' if tech_results.get('schema_markup', {}).get('pass') else 'No'
                            }

                            # Ensure competitor_data has the same columns for concatenation, fill missing with None/NaN
                            required_cols = list(target_metrics.keys())
                            competitor_data_aligned = competitor_data.reindex(columns=required_cols)

                            compare_df = pd.concat([pd.DataFrame([target_metrics]), competitor_data_aligned], ignore_index=True)
                            display_cols = ['Competitor', 'Title Length', 'Word Count', 'Images', 'Internal Links', 'Schema Markup?'] # Adjust if more metrics are fetched
                            st.dataframe(compare_df.set_index('Competitor')[display_cols], use_container_width=True)

                            # Add comparison charts only if data exists and can be made numeric
                            st.markdown("**Visual Comparison:**")
                            if 'Word Count' in compare_df.columns:
                                 chart_wc = create_comparison_bar_chart(compare_df.copy(), 'Word Count', 'Word Count Comparison')
                                 if chart_wc: st.altair_chart(chart_wc, use_container_width=True)
                            if 'Internal Links' in compare_df.columns:
                                 chart_il = create_comparison_bar_chart(compare_df.copy(), 'Internal Links', 'Internal Link Count Comparison')
                                 if chart_il: st.altair_chart(chart_il, use_container_width=True)


                            st.markdown("**SERP Features Analysis (Simulated or via SERP API):**")
                            if serp_features:
                                features_found = []
                                if serp_features.get('featured_snippet'): features_found.append("Featured Snippet")
                                if serp_features.get('related_questions'): features_found.append("People Also Ask")
                                if serp_features.get('knowledge_graph'): features_found.append("Knowledge Graph")
                                if serp_features.get('inline_videos'): features_found.append("Video Pack")
                                if serp_features.get('top_stories'): features_found.append("Top Stories")
                                if serp_features.get('local_pack'): features_found.append("Local Pack")

                                if features_found:
                                    st.write(f"- Features Detected: {', '.join(features_found)}")
                                else:
                                     st.write("- No major SERP features detected in simulation/API response.")
                            else:
                                st.write("SERP Feature data not available.")

                        else:
                            st.warning("Competitor data could not be generated or analyzed.")


                except Exception as e:
                    analysis_placeholder.error(f"An unexpected error occurred during analysis: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}") # Show full traceback for debugging

    # Add footer or additional info if desired
    st.sidebar.markdown("---")
    st.sidebar.info("App by [Your Name/Company] | Uses simulated data for competitors unless SERP API is configured.")


if __name__ == "__main__":
    main()
