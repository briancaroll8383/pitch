"""
profile_builder.py
------------------
Takes raw data collected by data_fetcher.py and uses Claude to synthesize
it into a structured 360° band profile.
"""

import json
import anthropic

PROFILE_SYSTEM_PROMPT = """You are a professional music industry analyst and band strategist.
Your job is to analyze raw data about a musical artist/band from multiple sources and
synthesize it into a comprehensive, structured 360° profile.

You ALWAYS respond with valid JSON. No markdown, no code blocks — pure JSON only.
Be analytical, specific, and actionable. Fill in gaps using your own music knowledge
when the provided data is sparse."""

PROFILE_USER_TEMPLATE = """
Here is raw data collected from multiple sources about the band/artist: "{band_name}"

Sources fetched: {sources_status}

{custom_biography_section}=== WIKIPEDIA ===
{wikipedia}

=== MUSICBRAINZ (discography + official links) ===
{musicbrainz}

=== ENCYCLOPEDIA METALLUM ===
{metallum}

=== SPOTIFY ===
{spotify}

=== LAST.FM ===
{lastfm}

=== SOCIAL LINKS FOUND ===
{social_links}

=== DUCKDUCKGO ===
{duckduckgo}

=== REDDIT ===
{reddit}

Some sources may show errors if credentials are not configured — use your own knowledge to fill gaps.
IMPORTANT: If Spotify data is available, populate spotify_followers and spotify_popularity from it.
If Last.fm data is available, populate lastfm_listeners and lastfm_playcount from it.
For known_accounts in social_presence, use the actual URLs from the SOCIAL LINKS FOUND section.
Based on ALL available data above (and your own knowledge), create a comprehensive 360° profile.
Return a JSON object with EXACTLY this structure:

{{
  "name": "band/artist name",
  "tagline": "one sentence that captures their essence",
  "overview": "3-4 sentence paragraph about who they are",
  "origin": {{
    "country": "",
    "city": "",
    "formed_year": "",
    "status": "active | hiatus | disbanded"
  }},
  "genre": {{
    "primary": "main genre",
    "secondary": ["sub-genre 1", "sub-genre 2"],
    "sound_description": "2-3 sentences describing their musical style"
  }},
  "members": [
    {{"name": "", "role": "", "joined": "", "left": null}}
  ],
  "discography": [
    {{
      "title": "",
      "type": "album | EP | single",
      "release_date": "YYYY-MM-DD or YYYY",
      "significance": "why this release matters"
    }}
  ],
  "milestones": [
    {{"year": "", "event": "significant milestone or achievement"}}
  ],
  "audience": {{
    "demographics": "description of typical fan demographics",
    "psychographics": "values, interests, lifestyle of fans",
    "community_vibe": "how the fanbase behaves and engages",
    "size_estimate": "niche | growing | established | mainstream"
  }},
  "social_presence": {{
    "platforms": ["list of platforms they're likely active on"],
    "content_style": "how they typically communicate with fans",
    "engagement_level": "low | medium | high | very high",
    "known_accounts": {{
      "instagram": "",
      "twitter": "",
      "facebook": "",
      "youtube": "",
      "spotify": "",
      "tiktok": ""
    }}
  }},
  "brand": {{
    "visual_identity": "description of their visual aesthetic",
    "themes": ["theme1", "theme2", "theme3"],
    "tone_of_voice": "how they speak — dark, playful, intellectual, raw, etc.",
    "associations": ["things the band is strongly associated with"]
  }},
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "opportunities": ["marketing opportunity 1", "opportunity 2", "opportunity 3"],
  "reddit_presence": {{
    "main_subreddit": "",
    "active_communities": ["r/subreddit1", "r/subreddit2"],
    "community_sentiment": "positive | mixed | negative | unknown"
  }},
  "upcoming_anniversaries": [
    {{
      "event": "e.g. 10th anniversary of Album X",
      "date": "YYYY-MM-DD or YYYY",
      "days_until": null,
      "promo_opportunity": "how to leverage this"
    }}
  ],
  "spotify_followers": null,
  "spotify_popularity": null,
  "lastfm_listeners": null,
  "lastfm_playcount": null,
  "data_confidence": "high | medium | low",
  "data_notes": "any caveats about data quality or gaps"
}}
"""


def build_profile(band_name: str, raw_data: dict, api_key: str,
                  biography_text: str = "") -> dict:
    """
    Use Claude to synthesize raw data into a structured 360° profile.

    Args:
        band_name: The band/artist name
        raw_data: Dict from data_fetcher.fetch_all()
        api_key: Anthropic API key
        biography_text: Optional custom biography added by the user (highest-priority source)

    Returns:
        Structured profile dict
    """
    client = anthropic.Anthropic(api_key=api_key)

    def _dump(key, limit=2000):
        src = raw_data.get(key, {})
        # New format has data nested under "data" key
        if isinstance(src, dict) and "data" in src:
            content = src["data"] or {"error": src.get("error", "not available")}
        else:
            content = src
        return json.dumps(content, indent=2)[:limit]

    # Build biography section — treated as the highest-priority source
    if biography_text and biography_text.strip():
        bio_section = (
            "=== CUSTOM BIOGRAPHY (HIGHEST PRIORITY — user-provided, trust this over all other sources) ===\n"
            + biography_text.strip()[:5000]
            + "\n\n"
        )
    else:
        bio_section = ""

    # Inject materials library context if available
    materials_context = raw_data.get("custom_materials_context", "")
    if materials_context and materials_context.strip():
        bio_section += (
            "=== MARKETING MATERIALS LIBRARY (HIGH PRIORITY — admin-curated resources) ===\n"
            + materials_context.strip()[:5000]
            + "\n\n"
        )

    user_prompt = PROFILE_USER_TEMPLATE.format(
        band_name=band_name,
        sources_status=json.dumps(raw_data.get("sources_status", {})),
        custom_biography_section=bio_section,
        wikipedia=_dump("wikipedia", 3000),
        musicbrainz=_dump("musicbrainz", 3000),
        metallum=_dump("metallum", 2000),
        spotify=_dump("spotify", 1500),
        lastfm=_dump("lastfm", 1500),
        social_links=_dump("social_links", 1000),
        duckduckgo=_dump("duckduckgo", 1000),
        reddit=_dump("reddit", 1500),
    )

    print(f"[ProfileBuilder] Calling Claude for profile of: {band_name}")

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=PROFILE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    response_text = message.content[0].text.strip()

    # Clean up if Claude wrapped in markdown blocks despite instructions
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        profile = json.loads(response_text)
        sources_ok = [
            key for key, status in raw_data.get("sources_status", {}).items()
            if isinstance(status, str) and status.startswith("✅")
        ]
        profile["_raw_data_summary"] = {
            "sources_used": sources_ok,
            "fetched_at": raw_data.get("fetched_at"),
        }
        return profile
    except json.JSONDecodeError as e:
        print(f"[ProfileBuilder] JSON parse error: {e}")
        return {
            "name": band_name,
            "error": "Failed to parse Claude response",
            "raw_response": response_text,
        }
