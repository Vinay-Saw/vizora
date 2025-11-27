"""
Unit tests for app.py solve_quiz function
Tests that email and secret are mandatory and validated
"""
import pytest
import os
import sys

# Set test environment variables before importing app
os.environ["SECRET_KEY"] = "test_secret_key"
os.environ["STUDENT_EMAIL"] = "test@example.com"

# Import the solve_quiz function
from app import solve_quiz


@pytest.mark.asyncio
async def test_solve_quiz_missing_email():
    """Test that empty email returns error"""
    result = await solve_quiz("https://example.com/quiz", "", "test_secret_key")
    assert "Email is required" in result


@pytest.mark.asyncio
async def test_solve_quiz_missing_secret():
    """Test that empty secret returns error"""
    result = await solve_quiz("https://example.com/quiz", "test@example.com", "")
    assert "Secret key is required" in result


@pytest.mark.asyncio
async def test_solve_quiz_whitespace_email():
    """Test that whitespace-only email returns error"""
    result = await solve_quiz("https://example.com/quiz", "   ", "test_secret_key")
    assert "Email is required" in result


@pytest.mark.asyncio
async def test_solve_quiz_whitespace_secret():
    """Test that whitespace-only secret returns error"""
    result = await solve_quiz("https://example.com/quiz", "test@example.com", "   ")
    assert "Secret key is required" in result


@pytest.mark.asyncio
async def test_solve_quiz_invalid_email():
    """Test that invalid email returns forbidden error"""
    result = await solve_quiz("https://example.com/quiz", "wrong@example.com", "test_secret_key")
    assert "Invalid email" in result


@pytest.mark.asyncio
async def test_solve_quiz_invalid_secret():
    """Test that invalid secret returns forbidden error"""
    result = await solve_quiz("https://example.com/quiz", "test@example.com", "wrong_secret")
    assert "Invalid secret" in result


@pytest.mark.asyncio
async def test_solve_quiz_none_email():
    """Test that None email returns error"""
    result = await solve_quiz("https://example.com/quiz", None, "test_secret_key")
    assert "Email is required" in result


@pytest.mark.asyncio
async def test_solve_quiz_none_secret():
    """Test that None secret returns error"""
    result = await solve_quiz("https://example.com/quiz", "test@example.com", None)
    assert "Secret key is required" in result
