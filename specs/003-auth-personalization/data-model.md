# Data Model: Authentication + Personalization + AI Summaries

## Overview
This document defines the data models for the authentication, personalization, and AI summaries feature. It specifies entity structures, relationships, validation rules, and state transitions.

## Entity: User
*Managed by BetterAuth*

**Fields**:
- `id` (string): Unique identifier for the user
- `email` (string): User's email address (primary login)
- `emailVerified` (datetime): Timestamp when email was verified
- `image` (string, nullable): URL to user's profile image
- `name` (string, nullable): User's display name

**Relationships**:
- One-to-One: UserProfile (via user_id foreign key)
- One-to-Many: PersonalizationRecord (via user_id foreign key)
- One-to-Many: AISummary (via user_id foreign key, for access tracking)

## Entity: UserProfile
*Custom table for technical background information*

**Fields**:
- `id` (UUID): Primary key
- `user_id` (string): Foreign key to BetterAuth user
- `python_experience` (enum): ["none", "beginner", "intermediate", "advanced", "expert"]
- `cpp_experience` (enum): ["none", "beginner", "intermediate", "advanced", "expert"]
- `js_ts_experience` (enum): ["none", "beginner", "intermediate", "advanced", "expert"]
- `ai_ml_familiarity` (enum): ["none", "beginner", "intermediate", "advanced", "expert"]
- `ros2_experience` (enum): ["none", "beginner", "intermediate", "advanced", "expert"]
- `gpu_details` (enum): ["none", "1650", "3050+", "4070+", "cloud_gpu"]
- `ram_capacity` (enum): ["4GB", "8GB", "16GB", "32GB", "64GB+"]
- `operating_system` (enum): ["linux", "windows", "mac"]
- `jetson_ownership` (boolean): Whether user owns Jetson hardware
- `realsense_lidar_availability` (boolean): Whether user has RealSense/LiDAR
- `created_at` (datetime): Timestamp when profile was created
- `updated_at` (datetime): Timestamp when profile was last updated

**Validation Rules**:
- `user_id` must reference an existing BetterAuth user
- All experience/comfort level fields must be one of the defined enum values
- Profile must be created during signup process

**Relationships**:
- Many-to-One: User (via user_id foreign key)
- One-to-Many: PersonalizationRecord (via user_id foreign key)

## Entity: PersonalizationRecord
*Stores personalized content adaptations for users and chapters*

**Fields**:
- `id` (UUID): Primary key
- `user_id` (string): Foreign key to BetterAuth user
- `chapter_id` (string): Identifier for the chapter being personalized
- `personalized_content` (text): The personalized version of the chapter content
- `personalization_metadata` (json): Additional data about the personalization (complexity level, examples used, etc.)
- `created_at` (datetime): Timestamp when personalization was created
- `updated_at` (datetime): Timestamp when personalization was last updated

**Validation Rules**:
- `user_id` must reference an existing BetterAuth user
- `chapter_id` must be a valid chapter identifier
- `personalized_content` must not exceed 10MB in size

**Relationships**:
- Many-to-One: User (via user_id foreign key)
- Many-to-One: UserProfile (via user_id foreign key)

## Entity: AISummary
*Cached AI-generated summaries for chapters*

**Fields**:
- `id` (UUID): Primary key
- `chapter_id` (string): Identifier for the chapter being summarized
- `summary_content` (text): The AI-generated summary
- `summary_metadata` (json): Additional data about the summary (model used, generation timestamp, etc.)
- `access_count` (integer): Number of times this summary has been accessed
- `last_accessed` (datetime): Timestamp of last access
- `created_at` (datetime): Timestamp when summary was first generated
- `updated_at` (datetime): Timestamp when summary was last regenerated

**Validation Rules**:
- `chapter_id` must be a valid chapter identifier
- `summary_content` must be between 100 and 10,000 characters
- `access_count` must be non-negative

**Relationships**:
- One-to-Many: User access logs (for tracking who has accessed summaries)

## Database Relationships Diagram

```
[User] 1 -----> 1 [UserProfile]
 |                  |
 |                  |  (user_id)
 |                  |
 | 1                | 1
 |                  |
 v                  v
[User] 1 -----> * [PersonalizationRecord]
 |
 | 1
 |
 v
[User] 1 -----> * [AISummary] (via access logs)
```

## Validation Rules Summary

### User Profile Validation
- All experience levels must be from the defined enums
- At least 50% of background survey fields must be completed during signup
- Email format validation is handled by BetterAuth
- No personally identifiable information beyond technical background

### Personalization Validation
- Personalization can only be generated for authenticated users
- Original chapter content integrity must be preserved in personalization process
- Personalization must be regenerated if user profile changes significantly

### AI Summary Validation
- AI summaries must be generated only for authenticated users
- Summary content must pass content safety checks
- Caching should have TTL to ensure content freshness

## State Transitions

### User Profile State Transitions
- `incomplete` → `complete` (when all required fields are filled during signup)
- `complete` → `updated` (when user modifies their profile)

### Personalization State Transitions
- `not_generated` → `generating` (when user requests personalization)
- `generating` → `generated` (when personalization is complete)
- `generated` → `regenerating` (when source content or user profile changes significantly)

### AI Summary State Transitions
- `not_generated` → `generating` (when first requested by authenticated user)
- `generating` → `cached` (when summary is generated and stored)
- `cached` → `expired` (when TTL expires or source content changes)
- `expired` → `regenerating` (when requested after expiration)