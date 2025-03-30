# schemas.py

import datetime
from typing import List, Optional, Literal, Dict, Any, TypedDict

# Import the V2 way of defining validators and the info object
from pydantic import BaseModel, Field, field_validator, ValidationInfo, ValidationError

# --- Backend Expected JSON Formats (Pydantic Models) ---


class RelatedPost(BaseModel):
    """Schema for related posts within a BlogPost."""

    title: str
    slug: str


class BlogPost(BaseModel):
    """Schema for the /blog endpoint."""

    slug: str = Field(..., description="URL-friendly identifier based on the title")
    title: str = Field(..., min_length=1)
    date: str = Field(..., description="Date in ISO format YYYY-MM-DD")
    excerpt: str = Field(..., min_length=1)
    coverImage: Optional[str] = Field(
        default="/default-cover.jpg", description="URL or path to cover image"
    )
    readingTime: str = Field(
        ..., description="Estimated reading time, e.g., '5 min read'"
    )
    tags: List[str] = Field(default_factory=list)
    content: str = Field(
        ..., description="Blog post content in Markdown format", min_length=1
    )
    relatedPosts: Optional[List[RelatedPost]] = Field(default_factory=list)

    # Use V2 field_validator for specific fields
    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str):
        """Ensures date is in YYYY-MM-DD format."""
        if not isinstance(v, str):
            raise ValueError("Date must be a string")  # Added type check
        try:
            datetime.datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

    @field_validator("readingTime")
    @classmethod
    def validate_reading_time_format(cls, v: str):
        """Ensures readingTime follows the 'X min read' format."""
        if not isinstance(v, str) or not v.endswith(" min read"):
            raise ValueError("readingTime must be a string ending with ' min read'")
        try:
            int(v.split(" ")[0])
        except (ValueError, IndexError):
            raise ValueError("readingTime must start with a number, e.g., '5 min read'")
        return v

    @field_validator("slug")
    @classmethod
    def validate_slug_format(cls, v: str):
        """Ensures slug is URL-friendly (basic check)."""
        if not isinstance(v, str):
            raise ValueError("Slug must be a string")  # Added type check
        if not v or any(c in v for c in [" ", "/", "?", "&", "#"]):
            raise ValueError(
                "Slug must be URL-friendly (no spaces or invalid URL characters like /, ?, &, #)"
            )
        return v


class WorkExperience(BaseModel):
    """Schema for the /experience/work endpoint."""

    title: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    location: Optional[str] = None
    startDate: str = Field(
        ..., description="Start date, recommend YYYY-MM or YYYY-MM-DD"
    )
    endDate: str = Field(..., description="End date (YYYY-MM, YYYY-MM-DD) or 'Present'")
    description: List[str] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="Bulleted list of responsibilities/achievements",
    )


class Education(BaseModel):
    """Schema for the /experience/education endpoint."""

    degree: str = Field(..., min_length=1)
    institution: str = Field(..., min_length=1)
    startYear: str = Field(..., description="Starting year (YYYY)")
    endYear: str = Field(..., description="Ending year (YYYY) or 'Present'")
    location: Optional[str] = None
    description: Optional[str] = None

    # ****** CORRECTED VALIDATOR USING Pydantic V2 SYNTAX ******
    @field_validator("startYear", "endYear")
    @classmethod
    def validate_year_format(
        cls, v: str, info: ValidationInfo
    ):  # Use info: ValidationInfo
        """Checks if the year is YYYY or allows 'Present' for endYear."""
        if not isinstance(v, str):
            raise ValueError("Year must be a string")  # Added type check

        # Access field name via info.field_name
        if info.field_name == "endYear" and v == "Present":
            return v
        if not (v.isdigit() and len(v) == 4):
            raise ValueError(
                f"{info.field_name} must be in YYYY format (or 'Present' for endYear)"
            )
        # Optional: Range check
        # try:
        #     year_int = int(v)
        #     current_year = datetime.datetime.now().year
        #     if not (1950 < year_int <= current_year + 5):
        #         raise ValueError("Year seems unrealistic")
        # except ValueError:
        #     # This case should be caught by the isdigit check above, but defensive check
        #      raise ValueError(f"{info.field_name} must be a valid year number")
        return v


class Achievement(BaseModel):
    """Schema for the /experience/achievement endpoint."""

    title: str = Field(..., min_length=1)
    organization: str = Field(..., min_length=1)
    date: str = Field(
        ..., description="Date of achievement, recommend YYYY-MM or YYYY-MM-DD"
    )
    description: str = Field(..., min_length=1)


class SkillCategory(BaseModel):
    """Schema for the /skills endpoint (represents one category)."""

    name: str = Field(
        ..., description="Category name, e.g., 'Programming Languages'", min_length=1
    )
    skills: List[str] = Field(
        ..., description="List of skills in this category", min_length=1
    )


# --- LangGraph State Definition ---

ClassificationType = Literal[
    "work-experience", "education", "achievement", "skill", "blog"
]


class AgentState(TypedDict):
    """Defines the structure of the state passed between LangGraph nodes."""

    # Input data from the monitor
    raw_post_data: Dict[
        str, Any
    ]  # Keys: 'id', 'text', 'media_url', 'timestamp' (ISO str)

    # Output of the Triage Agent
    classifications: List[ClassificationType]

    # Output of Transformation Agents: Maps classification type to validated Pydantic model(s)
    # For 'skill', the value is List[SkillCategory]. For others, it's the single model instance.
    transformed_data: Dict[ClassificationType, Any]

    # Output of the Publisher Agent: Maps published item identifier to its status
    publish_results: Dict[
        str, Dict[str, Any]
    ]  # e.g., {"blog_my-slug": {"status": 201}}

    # Accumulated error messages
    error_messages: List[str]
