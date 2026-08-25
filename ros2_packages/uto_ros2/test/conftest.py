"""Run package tests with paths relative to the ament package root."""
import os
from pathlib import Path


def pytest_sessionstart(session):
    os.chdir(Path(__file__).resolve().parents[1])
