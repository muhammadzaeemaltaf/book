# API Contracts: Authentication + Personalization + AI Summaries

## Overview
This document defines the API contracts for the authentication, personalization, and AI summaries feature. These contracts specify the endpoints, request/response formats, and error handling for all API interactions.

## Authentication Endpoints

### POST /api/auth/signup
Register a new user with background survey information.

**Request**:
```json
{
  "email": "user@example.com",
  "password": "securePassword123",
  "name": "John Doe",
  "background": {
    "python_experience": "intermediate",
    "cpp_experience": "beginner",
    "js_ts_experience": "advanced",
    "ai_ml_familiarity": "intermediate",
    "ros2_experience": "none",
    "gpu_details": "4070+",
    "ram_capacity": "32GB",
    "operating_system": "linux",
    "jetson_ownership": false,
    "realsense_lidar_availability": true
  }
}
```

**Response (201 Created)**:
```json
{
  "user": {
    "id": "user_123456789",
    "email": "user@example.com",
    "name": "John Doe"
  },
  "session_token": "session_token_abc123",
  "profile_created": true
}
```

**Errors**:
- 400: Invalid input data
- 409: Email already exists

### POST /api/auth/signin
Authenticate an existing user.

**Request**:
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response (200 OK)**:
```json
{
  "user": {
    "id": "user_123456789",
    "email": "user@example.com",
    "name": "John Doe"
  },
  "session_token": "session_token_abc123"
}
```

**Errors**:
- 400: Invalid input data
- 401: Invalid credentials

### POST /api/auth/signout
End the current user session.

**Request**: (No body required)
**Headers**: `Authorization: Bearer {session_token}`

**Response (200 OK)**:
```json
{
  "message": "Successfully signed out"
}
```

**Errors**:
- 401: Invalid or expired session

### GET /api/auth/me
Get information about the currently authenticated user.

**Headers**: `Authorization: Bearer {session_token}`

**Response (200 OK)**:
```json
{
  "user": {
    "id": "user_123456789",
    "email": "user@example.com",
    "name": "John Doe",
    "emailVerified": "2025-12-12T10:00:00Z"
  },
  "profile": {
    "python_experience": "intermediate",
    "cpp_experience": "beginner",
    "js_ts_experience": "advanced",
    "ai_ml_familiarity": "intermediate",
    "ros2_experience": "none",
    "gpu_details": "4070+",
    "ram_capacity": "32GB",
    "operating_system": "linux",
    "jetson_ownership": false,
    "realsense_lidar_availability": true,
    "created_at": "2025-12-12T10:00:00Z",
    "updated_at": "2025-12-12T10:00:00Z"
  }
}
```

**Errors**:
- 401: Invalid or expired session

### PUT /api/user/profile
Update user's technical background profile.

**Headers**: `Authorization: Bearer {session_token}`

**Request**:
```json
{
  "background": {
    "python_experience": "advanced",
    "cpp_experience": "intermediate",
    "ai_ml_familiarity": "advanced"
  }
}
```

**Response (200 OK)**:
```json
{
  "profile": {
    "id": "profile_987654321",
    "user_id": "user_123456789",
    "python_experience": "advanced",
    "cpp_experience": "intermediate",
    "js_ts_experience": "advanced",
    "ai_ml_familiarity": "advanced",
    "ros2_experience": "none",
    "gpu_details": "4070+",
    "ram_capacity": "32GB",
    "operating_system": "linux",
    "jetson_ownership": false,
    "realsense_lidar_availability": true,
    "created_at": "2025-12-12T10:00:00Z",
    "updated_at": "2025-12-12T11:00:00Z"
  }
}
```

**Errors**:
- 400: Invalid input data
- 401: Invalid or expired session

## Personalization Endpoints

### POST /api/personalize/{chapter_id}
Generate personalized content for a specific chapter based on user profile.

**Headers**: `Authorization: Bearer {session_token}`

**Response (200 OK)**:
```json
{
  "chapter_id": "module-01-ros2/chapter-1",
  "personalized_content": "# Personalized Chapter Content...\nBased on your advanced Python experience...",
  "metadata": {
    "complexity_level": "advanced",
    "adapted_for": ["python_experience", "ros2_experience"],
    "generation_timestamp": "2025-12-12T11:00:00Z"
  }
}
```

**Errors**:
- 401: Invalid or expired session
- 404: Chapter not found
- 422: User profile incomplete for personalization

## AI Summary Endpoints

### GET /api/summary/{chapter_id}
Get AI-generated summary for a specific chapter.

**Headers**: `Authorization: Bearer {session_token}`

**Response (200 OK)**:
```json
{
  "chapter_id": "module-01-ros2/chapter-1",
  "summary": "This chapter covers the fundamentals of ROS 2...",
  "metadata": {
    "model_used": "gemini",
    "generation_timestamp": "2025-12-12T10:30:00Z",
    "cached": true
  }
}
```

**Errors**:
- 401: Invalid or expired session
- 404: Chapter not found
- 503: AI service temporarily unavailable

## Common Error Responses

### 400 Bad Request
```json
{
  "error": "invalid_input",
  "message": "The request contains invalid data",
  "details": {
    "field": "email",
    "issue": "Invalid email format"
  }
}
```

### 401 Unauthorized
```json
{
  "error": "unauthorized",
  "message": "Authentication required or session expired"
}
```

### 404 Not Found
```json
{
  "error": "not_found",
  "message": "The requested resource was not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "internal_error",
  "message": "An unexpected error occurred"
}
```