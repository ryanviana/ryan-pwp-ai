# agents.py

import os
import httpx
import json
import asyncio
import datetime  # Needed for default year in transform_to_education
from typing import List, Dict, Any, Optional, cast

from dotenv import load_dotenv
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.exceptions import OutputParserException

# Import schemas and utilities (ensure these files exist and are in your PYTHONPATH)
from schemas import (
    ClassificationType,
    BlogPost,
    WorkExperience,
    Education,
    Achievement,
    SkillCategory,
)
from utils import (
    logger,
    estimate_reading_time,
    generate_slug,
    get_current_date_iso,
    validate_data,
)

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Read the model name, strip whitespace, remove comments, and remove quotes
raw_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
processed_name = raw_model_name.split("#")[0].strip()
if (processed_name.startswith('"') and processed_name.endswith('"')) or (
    processed_name.startswith("'") and processed_name.endswith("'")
):
    OPENAI_MODEL_NAME = processed_name[1:-1]
else:
    OPENAI_MODEL_NAME = processed_name
# Also parse the BACKEND_BASE_URL like the model name
raw_backend_url = os.getenv("BACKEND_BASE_URL", "")
processed_url = raw_backend_url.split("#")[0].strip()
if (processed_url.startswith('"') and processed_url.endswith('"')) or (
    processed_url.startswith("'") and processed_url.endswith("'")
):
    BACKEND_BASE_URL = processed_url[1:-1]
else:
    BACKEND_BASE_URL = processed_url

# --- Basic Validation ---
if not OPENAI_API_KEY:
    logger.critical("OPENAI_API_KEY environment variable not set.")
    raise ValueError("Missing OpenAI API Key")
if not BACKEND_BASE_URL:
    logger.critical("BACKEND_BASE_URL environment variable not set.")
    raise ValueError("Missing Backend Base URL")
# Validate the processed model name
if not OPENAI_MODEL_NAME:
    OPENAI_MODEL_NAME = "gpt-4o"  # Fallback if processing results in empty string
    logger.warning(
        f"OPENAI_MODEL_NAME was empty after processing, defaulting to {OPENAI_MODEL_NAME}."
    )
else:
    logger.info(f"Using OpenAI Model: {OPENAI_MODEL_NAME}")

# --- Initialize LLM ---
try:
    llm = ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0.1,  # Lower temperature for more deterministic JSON output
        max_tokens=3000,  # Adjust based on expected output size
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    logger.info(f"Initialized LLM: {OPENAI_MODEL_NAME}")
except Exception as e:
    logger.critical(f"Failed to initialize OpenAI LLM: {e}", exc_info=True)
    raise


# --- Output Parsers ---
class ClassificationsList(BaseModel):
    """Schema for parsing the list of classifications from the Triage Agent."""

    classifications: List[ClassificationType] = Field(
        description="List of identified classifications (e.g., ['blog', 'work-experience'])"
    )


triage_parser = PydanticOutputParser(pydantic_object=ClassificationsList)


# --- LangGraph State Definition ---
class AgentState(TypedDict):
    raw_post_data: Dict[str, Any]
    classifications: List[ClassificationType]
    transformed_data: Dict[str, Any]
    publish_results: Dict[str, Any]
    error_messages: List[str]


# --- Agent Node Implementations ---


async def triage_agent(state: AgentState) -> Dict[str, Any]:
    """Classifies the LinkedIn post content. Node Function."""
    logger.info("--- Node: Triage Agent ---")
    errors = state.get("error_messages", [])
    classifications = []

    try:
        raw_post = state["raw_post_data"]
        post_content = raw_post.get("text", "").strip()
        if not post_content:
            logger.warning("Triage Agent: Raw post content is empty.")
            errors.append("Triage Error: Empty post content")
            return {"classifications": [], "error_messages": errors}

        # Use triple quotes and double braces to escape JSON examples
        system_message_text = """You are an expert classifier specializing in LinkedIn content. Your task is to analyze the provided LinkedIn post text and determine which predefined categories it belongs to.

**Available Categories:**
- work-experience: Posts announcing a new job, promotion, work anniversary, significant project completion, or discussing professional responsibilities.
- education: Posts mentioning completion of a course, degree, certification, workshop, or any formal/informal learning program.
- achievement: Posts highlighting awards, recognitions, patents, speaking engagements, publications, or significant personal/professional milestones NOT directly tied to standard job duties described elsewhere.
- skill: Posts explicitly focusing on learning, mastering, or utilizing specific technical skills (e.g., Python, AWS, Prompt Engineering), tools (e.g., Figma, Jira), or methodologies (e.g., Agile, Scrum). Often part of a broader post.
- blog: Posts sharing insights, reflections, opinions, or tutorials, OR acting as an announcement/summary for longer external content. Think: does this read like a mini-article?

**Instructions:**
1. Carefully read the LinkedIn post content.
2. Consider the *style* and *purpose* of the post. Does it aim to inform, teach, reflect, or announce something substantial?
3. Identify ALL categories that accurately describe the *main topics* of the post. A post can belong to multiple categories.
4. Return ONLY a JSON object containing a single key 'classifications' which holds a list of the identified category strings.
5. If no categories apply, return an empty list: {{"classifications": []}}

**Example Input Post (Blog style):**
"Por que tem um cronômetro gigante na minha sala?\n\nUma tarefa tende a se estender até o prazo que você dá para ela. Quanto maior a deadline, maior o tempo que você vai levar para completá-la. Nós estamos chamando isso de Mentalidade de Hackathon!"

**Example Output (Blog style):**
{{"classifications": ["blog"]}}

**Example Input Post (Work/Skill):**
"Thrilled to share I've started a new role as Lead AI Engineer at Innovatech! Excited to lead the development of cutting-edge AI solutions using Python, TensorFlow, and LangGraph. #AI #Engineering"

**Example Output (Work/Skill):**
{{"classifications": ["work-experience", "skill"]}}

**Output Format:**
Return a JSON object with a single key 'classifications' containing an array of category strings.
"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message_text),
                ("user", "LinkedIn Post:\n---\n{content}\n---\n\nClassify this post."),
            ]
        )

        logger.debug(f"Prompt expected input variables: {prompt.input_variables}")
        chain = prompt | llm | triage_parser

        # Supply only the "content" variable, as our prompt now expects that.
        input_data = {"content": post_content}
        try:
            result = await chain.ainvoke(input_data)
            potential_classifications = result.classifications
            valid_classification_values = list(ClassificationType.__args__)
            valid_classifications = [
                c for c in potential_classifications if c in valid_classification_values
            ]
            if len(valid_classifications) != len(potential_classifications):
                invalid_found = set(potential_classifications) - set(
                    valid_classifications
                )
                logger.warning(
                    f"Triage parser returned invalid classification values: {list(invalid_found)}. Keeping only valid ones: {valid_classifications}"
                )
            classifications = valid_classifications
            logger.info(f"Triage Agent Classification: {classifications}")
        except KeyError as e:
            logger.critical(
                f"Triage Agent Input KeyError during invoke - UNEXPECTED! Input: {input_data}. Error: {e}",
                exc_info=True,
            )
            errors.append(f"Triage Unexpected Input KeyError: {str(e)}")
            classifications = []
        except OutputParserException as e:
            logger.error(f"Triage Agent Output Parsing Error: {e}", exc_info=True)
            errors.append(f"Triage Parsing Error: {str(e)}")
            classifications = []
        except Exception as e:
            logger.error(f"Triage Agent Error during invoke/parse: {e}", exc_info=True)
            errors.append(f"Triage Error: {str(e)}")
            classifications = []
    except Exception as e:
        logger.error(f"Triage Agent Setup Error: {e}", exc_info=True)
        errors.append(f"Triage Agent Setup Error: {str(e)}")
        classifications = []

    return {"classifications": classifications, "error_messages": errors}


async def transform_to_blog(state: AgentState) -> Dict[str, Any]:
    """Transforms LinkedIn post content into a BlogPost JSON. Node Function."""
    node_name = "Transform Blog"
    logger.info(f"--- Node: {node_name} ---")
    transformed_data = state.get("transformed_data", {})
    errors = state.get("error_messages", [])
    try:
        raw_post = state["raw_post_data"]
        post_content = raw_post.get("text", "")
        media_url = raw_post.get("media_url")
        current_date = get_current_date_iso()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are an expert technical writer and content enhancer. Convert the provided LinkedIn post into a structured blog post for a personal website, adhering strictly to the requested JSON format.

**Task:** Generate a JSON object based on the LinkedIn post.

**JSON Requirements:**
- `slug`: Generate a URL-friendly slug from the title (lowercase, hyphens for spaces, max 80 chars).
- `title`: Create a catchy, relevant blog post title based on the post content.
- `date`: Use today's date: "{current_date}".
- `excerpt`: Write a concise 1-2 sentence summary of the blog post.
- `coverImage`: Use the provided media URL if available ("{media_url or 'None'}"), otherwise use "/default-cover.jpg".
- `readingTime`: Estimate reading time based *only* on the generated 'content' field (format: 'X min read'). If unsure, use '3 min read'.
- `tags`: Extract or generate 4-6 relevant keywords/tags (e.g., ["AI", "Python", "LangGraph"]).
- `content`: Rewrite the LinkedIn post into engaging blog content using Markdown. Include:
    - A brief, engaging introduction.
    - Use Markdown headings (`##`, `###`), lists (`*`, `-`), bold (`**text**`), italics (`*text*`).
    - Expand slightly on the original points if appropriate, maintaining the core message.
    - Ensure good structure and flow.
- `relatedPosts`: Leave as an empty list `[]`.

**Output Format:** Return ONLY the JSON object.
""",
                ),
                (
                    "user",
                    "LinkedIn Post:\n---\n{linkedin_post}\n---\n\nGenerate the blog post JSON.",
                ),
            ]
        )
        chain = prompt | llm | JsonOutputParser()
        result_json = await chain.ainvoke({"linkedin_post": post_content})
        if isinstance(result_json, dict):
            if "content" in result_json and "readingTime" not in result_json:
                result_json["readingTime"] = estimate_reading_time(
                    result_json["content"]
                )
            if "title" in result_json and (
                "slug" not in result_json or not result_json["slug"]
            ):
                result_json["slug"] = generate_slug(result_json["title"])
            result_json["date"] = result_json.get("date", current_date)
            result_json["coverImage"] = result_json.get(
                "coverImage", media_url or "/default-cover.jpg"
            )
            result_json["tags"] = result_json.get("tags", [])
            result_json["excerpt"] = result_json.get(
                "excerpt", result_json.get("content", "")[:150] + "..."
            )
            result_json["relatedPosts"] = result_json.get("relatedPosts", [])
            validated_data, error_msg = validate_data(result_json, BlogPost)
            if validated_data:
                transformed_data["blog"] = validated_data
                logger.info(f"{node_name}: Transformation successful.")
            else:
                logger.error(
                    f"{node_name} Validation Error: {error_msg}. LLM Output: {json.dumps(result_json)}"
                )
                errors.append(f"{node_name} Validation Error: {error_msg}")
        else:
            logger.error(
                f"{node_name}: LLM did not return a valid JSON object. Output: {result_json}"
            )
            errors.append(f"{node_name} Error: Invalid JSON output from LLM.")
    except OutputParserException as e:
        logger.error(f"{node_name} Output Parsing Error: {e}", exc_info=True)
        errors.append(f"{node_name} Parsing Error: {str(e)}")
    except Exception as e:
        logger.error(f"{node_name} Error: {e}", exc_info=True)
        errors.append(f"{node_name} Error: {str(e)}")
    return {"transformed_data": transformed_data, "error_messages": errors}


async def transform_to_work_experience(state: AgentState) -> Dict[str, Any]:
    """Transforms LinkedIn post content into a WorkExperience JSON. Node Function."""
    node_name = "Transform Work Experience"
    logger.info(f"--- Node: {node_name} ---")
    transformed_data = state.get("transformed_data", {})
    errors = state.get("error_messages", [])
    try:
        raw_post = state["raw_post_data"]
        post_content = raw_post.get("text", "")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert data extractor focused on professional work history. Extract work experience details from the LinkedIn post and format them strictly as JSON.

**Task:** Generate a JSON object representing the work experience.

**JSON Requirements:**
- `title`: The job title mentioned (e.g., "Software Engineer").
- `company`: The company name mentioned.
- `location`: The location (City, State, Country) if mentioned, otherwise null.
- `startDate`: The start date mentioned (e.g., "YYYY-MM", "YYYY-MM-DD", "Month YYYY"). If not specified, try to infer or use null.
- `endDate`: The end date. Use "Present" if it's a new role announcement or ongoing role. Use format like "YYYY-MM", "YYYY-MM-DD", "Month YYYY" if specified. Use null if not applicable.
- `description`: A list of strings summarizing key responsibilities or achievements mentioned. Create 1-4 concise bullet points. If no details are given, provide a single bullet based on the title, like ["Assumed the role of [Job Title] at [Company]."].

**Output Format:** Return ONLY the JSON object.
""",
                ),
                (
                    "user",
                    "LinkedIn Post:\n---\n{linkedin_post}\n---\n\nExtract work experience details into the specified JSON format.",
                ),
            ]
        )
        chain = prompt | llm | JsonOutputParser()
        result_json = await chain.ainvoke({"linkedin_post": post_content})
        if isinstance(result_json, dict):
            if "description" not in result_json or not result_json.get("description"):
                title = result_json.get("title", "the role")
                company = result_json.get("company", "the company")
                result_json["description"] = [f"Assumed {title} at {company}."]
            if "endDate" not in result_json and (
                "start" in post_content.lower() or "join" in post_content.lower()
            ):
                result_json["endDate"] = "Present"
            validated_data, error_msg = validate_data(result_json, WorkExperience)
            if validated_data:
                transformed_data["work-experience"] = validated_data
                logger.info(f"{node_name}: Transformation successful.")
            else:
                logger.error(
                    f"{node_name} Validation Error: {error_msg}. LLM Output: {json.dumps(result_json)}"
                )
                errors.append(f"{node_name} Validation Error: {error_msg}")
        else:
            logger.error(
                f"{node_name}: LLM did not return a valid JSON object. Output: {result_json}"
            )
            errors.append(f"{node_name} Error: Invalid JSON output from LLM.")
    except OutputParserException as e:
        logger.error(f"{node_name} Output Parsing Error: {e}", exc_info=True)
        errors.append(f"{node_name} Parsing Error: {str(e)}")
    except Exception as e:
        logger.error(f"{node_name} Error: {e}", exc_info=True)
        errors.append(f"{node_name} Error: {str(e)}")
    return {"transformed_data": transformed_data, "error_messages": errors}


async def transform_to_education(state: AgentState) -> Dict[str, Any]:
    """Transforms LinkedIn post content into an Education JSON. Node Function."""
    node_name = "Transform Education"
    logger.info(f"--- Node: {node_name} ---")
    transformed_data = state.get("transformed_data", {})
    errors = state.get("error_messages", [])
    try:
        raw_post = state["raw_post_data"]
        post_content = raw_post.get("text", "")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert data extractor focused on education history. Extract education details from the LinkedIn post and format them strictly as JSON.

**Task:** Generate a JSON object representing the education entry.

**JSON Requirements:**
- `degree`: The name of the degree, certificate, or course (e.g., "B.S. Computer Science", "Machine Learning Specialization").
- `institution`: The name of the institution (e.g., "Stanford University", "Coursera").
- `startYear`: The starting year (YYYY) if mentioned, otherwise null.
- `endYear`: The ending/completion year (YYYY). If the post implies recent completion, use the current year. If ongoing, use "Present". If not mentioned, use null.
- `location`: The location (City, State) if mentioned, otherwise null.
- `description`: A brief description if provided in the post, otherwise null.

**Output Format:** Return ONLY the JSON object.
""",
                ),
                (
                    "user",
                    "LinkedIn Post:\n---\n{linkedin_post}\n---\n\nExtract education details into the specified JSON format.",
                ),
            ]
        )
        chain = prompt | llm | JsonOutputParser()
        result_json = await chain.ainvoke({"linkedin_post": post_content})
        if isinstance(result_json, dict):
            if result_json.get("endYear") is None and any(
                word in post_content.lower()
                for word in ["completed", "finished", "earned", "received"]
            ):
                result_json["endYear"] = str(datetime.datetime.now().year)
            validated_data, error_msg = validate_data(result_json, Education)
            if validated_data:
                transformed_data["education"] = validated_data
                logger.info(f"{node_name}: Transformation successful.")
            else:
                logger.error(
                    f"{node_name} Validation Error: {error_msg}. LLM Output: {json.dumps(result_json)}"
                )
                errors.append(f"{node_name} Validation Error: {error_msg}")
        else:
            logger.error(
                f"{node_name}: LLM did not return a valid JSON object. Output: {result_json}"
            )
            errors.append(f"{node_name} Error: Invalid JSON output from LLM.")
    except OutputParserException as e:
        logger.error(f"{node_name} Output Parsing Error: {e}", exc_info=True)
        errors.append(f"{node_name} Parsing Error: {str(e)}")
    except Exception as e:
        logger.error(f"{node_name} Error: {e}", exc_info=True)
        errors.append(f"{node_name} Error: {str(e)}")
    return {"transformed_data": transformed_data, "error_messages": errors}


async def transform_to_achievement(state: AgentState) -> Dict[str, Any]:
    """Transforms LinkedIn post content into an Achievement JSON. Node Function."""
    node_name = "Transform Achievement"
    logger.info(f"--- Node: {node_name} ---")
    transformed_data = state.get("transformed_data", {})
    errors = state.get("error_messages", [])
    try:
        raw_post = state["raw_post_data"]
        post_content = raw_post.get("text", "")
        current_date = get_current_date_iso()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are an expert data extractor focused on achievements and recognitions. Extract achievement details (awards, publications, patents, speaking roles, significant milestones) from the LinkedIn post and format them strictly as JSON.

**Task:** Generate a JSON object representing the achievement.

**JSON Requirements:**
- `title`: The name or title of the achievement (e.g., "Received Top Voice Award", "Published Paper on AI Ethics", "Granted Patent for X").
- `organization`: The awarding organization, publication venue, or context (e.g., "LinkedIn", "IEEE Conference", "USPTO").
- `date`: The date or year the achievement occurred (e.g., "YYYY-MM-DD", "Month YYYY", "YYYY"). If not specified, use the current date: {current_date}.
- `description`: A brief description summarizing the achievement.

**Output Format:** Return ONLY the JSON object.
""",
                ),
                (
                    "user",
                    "LinkedIn Post:\n---\n{linkedin_post}\n---\n\nExtract achievement details into the specified JSON format.",
                ),
            ]
        )
        chain = prompt | llm | JsonOutputParser()
        result_json = await chain.ainvoke({"linkedin_post": post_content})
        if isinstance(result_json, dict):
            if "date" not in result_json or not result_json["date"]:
                result_json["date"] = current_date
            validated_data, error_msg = validate_data(result_json, Achievement)
            if validated_data:
                transformed_data["achievement"] = validated_data
                logger.info(f"{node_name}: Transformation successful.")
            else:
                logger.error(
                    f"{node_name} Validation Error: {error_msg}. LLM Output: {json.dumps(result_json)}"
                )
                errors.append(f"{node_name} Validation Error: {error_msg}")
        else:
            logger.error(
                f"{node_name}: LLM did not return a valid JSON object. Output: {result_json}"
            )
            errors.append(f"{node_name} Error: Invalid JSON output from LLM.")
    except OutputParserException as e:
        logger.error(f"{node_name} Output Parsing Error: {e}", exc_info=True)
        errors.append(f"{node_name} Parsing Error: {str(e)}")
    except Exception as e:
        logger.error(f"{node_name} Error: {e}", exc_info=True)
        errors.append(f"{node_name} Error: {str(e)}")
    return {"transformed_data": transformed_data, "error_messages": errors}


async def transform_to_skill(state: AgentState) -> Dict[str, Any]:
    """Transforms LinkedIn post content into a list of SkillCategory JSONs. Node Function."""
    node_name = "Transform Skill"
    logger.info(f"--- Node: {node_name} ---")
    transformed_data = state.get("transformed_data", {})
    errors = state.get("error_messages", [])
    validated_skill_categories = []
    try:
        raw_post = state["raw_post_data"]
        post_content = raw_post.get("text", "")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert skill extractor and categorizer. Identify specific skills (technical, software, tools, methodologies, languages, platforms, frameworks, soft skills) mentioned in the LinkedIn post.

**Task:** Generate a JSON object containing a list of skill categories with their associated skills.

**JSON Requirements:**
- The top-level JSON object must have a single key: "skill_categories".
- The value of "skill_categories" must be a list ([]).
- Each element in the list must be an object representing a skill category, with two keys:
    - `name`: A string representing the category name (e.g., "Programming Languages", "Cloud Platforms", "Project Management", "Soft Skills", "Software & Tools").
    - `skills`: A list of strings, where each string is a specific skill identified within that category (e.g., ["Python", "JavaScript"], ["AWS", "Azure"], ["Agile", "Scrum"], ["Communication", "Leadership"], ["Figma", "Docker"]).
- Group related skills under the most relevant category.
- If no specific skills are clearly mentioned, return {"skill_categories": []}.

**Example Output:**
```json
{
  "skill_categories": [
    { "name": "Programming Languages", "skills": ["Python", "TypeScript"] },
    { "name": "AI/ML Frameworks", "skills": ["TensorFlow", "LangGraph"] },
    { "name": "Cloud Platforms", "skills": ["AWS"] }
  ]
}
```

**Output Format:** Return ONLY the JSON object described above.
""",
                ),
                (
                    "user",
                    "LinkedIn Post:\n---\n{linkedin_post}\n---\n\nExtract and categorize skills into the specified JSON format.",
                ),
            ]
        )
        chain = prompt | llm | JsonOutputParser()
        result_json = await chain.ainvoke({"linkedin_post": post_content})

        if (
            isinstance(result_json, dict)
            and "skill_categories" in result_json
            and isinstance(result_json["skill_categories"], list)
        ):
            skill_categories_raw = result_json["skill_categories"]
            logger.debug(
                f"{node_name}: LLM returned {len(skill_categories_raw)} raw skill categories."
            )

            for i, category_data in enumerate(skill_categories_raw):
                if isinstance(category_data, dict):
                    validated_category, error_msg = validate_data(
                        category_data, SkillCategory
                    )
                    if validated_category:
                        validated_skill_categories.append(validated_category)
                    else:
                        logger.error(
                            f"{node_name} Validation Error (Category {i+1}): {error_msg}. Data: {json.dumps(category_data)}"
                        )
                        errors.append(
                            f"{node_name} Validation Error (Category {i+1}): {error_msg}"
                        )
                else:
                    logger.error(
                        f"{node_name}: Format Error: Item {i+1} in 'skill_categories' list is not a dictionary. Item: {category_data}"
                    )
                    errors.append(
                        f"{node_name} Format Error: Item {i+1} in 'skill_categories' not a dict"
                    )

            if validated_skill_categories:
                transformed_data["skill"] = validated_skill_categories
                logger.info(
                    f"{node_name}: Transformation successful. Found {len(validated_skill_categories)} valid skill categories."
                )
            elif not errors and skill_categories_raw:
                errors.append(
                    f"{node_name} Error: LLM provided categories, but none passed validation."
                )
            elif not skill_categories_raw:
                logger.info(
                    f"{node_name}: No skill categories identified or returned by LLM."
                )
        else:
            logger.error(
                f"{node_name}: LLM did not return a valid JSON object with 'skill_categories' list. Output: {result_json}"
            )
            errors.append(f"{node_name} Error: Invalid JSON structure from LLM.")
    except OutputParserException as e:
        logger.error(f"{node_name} Output Parsing Error: {e}", exc_info=True)
        errors.append(f"{node_name} Parsing Error: {str(e)}")
    except Exception as e:
        logger.error(f"{node_name} Error: {e}", exc_info=True)
        errors.append(f"{node_name} Error: {str(e)}")

    return {"transformed_data": transformed_data, "error_messages": errors}


# --- Content Publisher Agent ---
ENDPOINT_MAP = {
    "blog": {"endpoint": "/blog", "schema": BlogPost},
    "work-experience": {"endpoint": "/experience/work", "schema": WorkExperience},
    "education": {"endpoint": "/experience/education", "schema": Education},
    "achievement": {"endpoint": "/experience/achievement", "schema": Achievement},
    "skill": {"endpoint": "/skills", "schema": SkillCategory},
}


async def content_publisher_agent(state: AgentState) -> Dict[str, Any]:
    """Publishes the transformed and validated content to the backend API. Node Function."""
    node_name = "Content Publisher"
    logger.info(f"--- Node: {node_name} ---")
    transformed_items = state.get("transformed_data", {})
    publish_results = state.get("publish_results", {})
    errors = state.get("error_messages", [])

    if not transformed_items:
        logger.info(f"{node_name}: No transformed data available to publish.")
        return {"publish_results": publish_results, "error_messages": errors}

    headers = {"Content-Type": "application/json"}
    logger.debug(f"{node_name}: Using Backend Base URL: '{BACKEND_BASE_URL}'")
    async with httpx.AsyncClient(
        base_url=BACKEND_BASE_URL, timeout=30.0, follow_redirects=True, headers=headers
    ) as client:
        for content_type_str, data_to_publish in transformed_items.items():
            content_type: ClassificationType = cast(
                ClassificationType, content_type_str
            )
            if content_type not in ENDPOINT_MAP:
                logger.warning(
                    f"{node_name}: No endpoint mapping found for content type '{content_type}'. Skipping."
                )
                continue

            endpoint_info = ENDPOINT_MAP[content_type]
            endpoint = endpoint_info["endpoint"]
            expected_schema = endpoint_info["schema"]
            logger.debug(
                f"{node_name}: Preparing to publish {content_type} to {endpoint}"
            )

            items_to_process = []
            if content_type == "skill" and isinstance(data_to_publish, list):
                items_to_process = [
                    item
                    for item in data_to_publish
                    if isinstance(item, expected_schema)
                ]
                if len(items_to_process) != len(data_to_publish):
                    logger.warning(
                        f"{node_name}: Found non-{expected_schema.__name__} items in list for {content_type}. Skipping invalid items."
                    )
            elif isinstance(data_to_publish, expected_schema):
                items_to_process.append(data_to_publish)
            else:
                logger.error(
                    f"{node_name}: Data for {content_type} is not of expected type {expected_schema}. Got {type(data_to_publish)}. Skipping."
                )
                errors.append(
                    f"Publisher Error: Invalid data type for '{content_type}' ({type(data_to_publish)})"
                )
                continue

            if not items_to_process:
                logger.info(
                    f"{node_name}: No valid items to process for {content_type} after filtering."
                )
                continue

            item_publish_results = []
            for item_index, item in enumerate(items_to_process):
                try:
                    item_payload = item.model_dump_json()
                    item_payload_dict = json.loads(item_payload)
                    logger.info(
                        f"{node_name}: Sending POST for item {item_index+1} for {content_type} to {endpoint}"
                    )
                    logger.debug(
                        f"Payload for item {item_index+1}: {json.dumps(item_payload_dict, indent=2)}"
                    )

                    response = await client.post(endpoint, json=item_payload_dict)
                    if response.status_code in [200, 201]:
                        logger.info(
                            f"{node_name}: Successfully published item {item_index+1} for {content_type}. Status: {response.status_code}"
                        )
                        item_publish_results.append(
                            {
                                "status": response.status_code,
                                "response": response.text[:250],
                            }
                        )
                    else:
                        logger.error(
                            f"{node_name}: Failed to publish item {item_index+1} for {content_type}. Status: {response.status_code}, Response: {response.text}"
                        )
                        error_detail = f"Publish Error (item {item_index+1} to {endpoint}): Status {response.status_code} - {response.text[:250]}"
                        errors.append(error_detail)
                        item_publish_results.append(
                            {
                                "status": response.status_code,
                                "error": error_detail,
                            }
                        )
                except httpx.RequestError as e:
                    logger.error(
                        f"{node_name}: HTTP Request Error for {content_type} item {item_index+1} to {endpoint}: {e}",
                        exc_info=True,
                    )
                    error_detail = f"Publish Request Error (item {item_index+1}): {e}"
                    errors.append(error_detail)
                    item_publish_results.append(
                        {
                            "status": "Network Error",
                            "error": error_detail,
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"{node_name}: Unexpected error publishing item {item_index+1} for {content_type}: {e}",
                        exc_info=True,
                    )
                    error_detail = (
                        f"Publish Unexpected Error (item {item_index+1}): {e}"
                    )
                    errors.append(error_detail)
                    item_publish_results.append(
                        {
                            "status": "Unexpected Error",
                            "error": error_detail,
                        }
                    )
            publish_results[content_type_str] = item_publish_results

    return {"publish_results": publish_results, "error_messages": errors}
