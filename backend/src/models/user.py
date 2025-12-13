"""
User and UserProfile models for the authentication and personalization feature.
"""
from datetime import datetime
from typing import Optional, List
import uuid

from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import JSON
from enum import Enum


class ExperienceLevel(str, Enum):
    none = "none"
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"


class GPUOption(str, Enum):
    none = "none"
    gtx_1650 = "1650"
    gtx_3050_plus = "3050+"
    rtx_4070_plus = "4070+"
    cloud_gpu = "cloud_gpu"


class RAMCapacity(str, Enum):
    gb_4 = "4GB"
    gb_8 = "8GB"
    gb_16 = "16GB"
    gb_32 = "32GB"
    gb_64_plus = "64GB+"


class OperatingSystem(str, Enum):
    linux = "linux"
    windows = "windows"
    mac = "mac"


class User(SQLModel, table=True):
    """
    User model - managed by BetterAuth.
    This is a reference model to understand the structure.
    """
    __tablename__ = "users"  # Explicitly set table name

    id: str = Field(primary_key=True)
    email: str = Field(unique=True, index=True)
    email_verified: Optional[datetime] = None
    image: Optional[str] = None
    name: Optional[str] = None

    # Relationship to user profile
    profile: Optional["UserProfile"] = Relationship(
        back_populates="user",
        sa_relationship_kwargs={"uselist": False}
    )

    # Relationship to personalization records
    personalization_records: list["PersonalizationRecord"] = Relationship(
        back_populates="user"
    )


class UserProfile(SQLModel, table=True):
    """
    Custom table for technical background information.
    """
    __tablename__ = "user_profiles"  # Explicitly set table name

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: str = Field(foreign_key="users.id", unique=True)  # Fixed: referencing correct table name

    # Programming experience
    python_experience: ExperienceLevel = Field(default=ExperienceLevel.none)
    cpp_experience: ExperienceLevel = Field(default=ExperienceLevel.none)
    js_ts_experience: ExperienceLevel = Field(default=ExperienceLevel.none)

    # AI/ML familiarity
    ai_ml_familiarity: ExperienceLevel = Field(default=ExperienceLevel.none)

    # Robotics experience
    ros2_experience: ExperienceLevel = Field(default=ExperienceLevel.none)

    # Hardware details
    gpu_details: GPUOption = Field(default=GPUOption.none)
    ram_capacity: RAMCapacity = Field(default=RAMCapacity.gb_4)
    operating_system: OperatingSystem = Field(default=OperatingSystem.linux)

    # Equipment ownership
    jetson_ownership: bool = Field(default=False)
    realsense_lidar_availability: bool = Field(default=False)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship to user
    user: User = Relationship(back_populates="profile")


class PersonalizationRecord(SQLModel, table=True):
    """
    Stores personalized content adaptations for users and chapters.
    """
    __tablename__ = "personalization_records"  # Explicitly set table name

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: str = Field(foreign_key="users.id")  # Fixed: referencing correct table name
    chapter_id: str = Field(index=True)

    personalized_content: str = Field(max_length=10 * 1024 * 1024)  # 10MB max
    personalization_metadata: dict = Field(default_factory=dict, sa_type=JSON)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    user: User = Relationship(back_populates="personalization_records")


class AISummary(SQLModel, table=True):
    """
    Cached AI-generated summaries for chapters.
    """
    __tablename__ = "ai_summaries"  # Explicitly set table name

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    chapter_id: str = Field(index=True)

    summary_content: str = Field(min_length=100, max_length=10000)
    summary_metadata: dict = Field(default_factory=dict, sa_type=JSON)

    access_count: int = Field(default=0, ge=0)
    last_accessed: Optional[datetime] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Add index for performance
    __table_args__ = (
        # Additional constraints can be added here
    )