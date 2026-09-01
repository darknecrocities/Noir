"""Unit test for automated plain-text ELI5 report generator."""

from pathlib import Path
import pytest

from noir.reporting.report_generator import TrainingReportGenerator


def test_report_generator_content(tmp_path):
    report_text = TrainingReportGenerator.generate_markdown_report(
        experiment_id="test_exp_123",
        step=50,
        epoch=2,
        metrics={"train_loss": 3.85, "perplexity": 47.2, "val_perplexity": 52.1},
        emotion_state={"curiosity": 0.85, "confidence": 0.72, "surprise": 0.15, "persistence": 0.90},
        resources=[
            {
                "title": "Quantum Computing Fundamentals",
                "url": "https://en.wikipedia.org/wiki/Quantum_computing",
                "source_type": "Wikipedia",
                "character_count": 1500,
            }
        ],
        generated_sample="The future of intelligence is expanding rapidly.",
        device_name="NVIDIA GeForce RTX 3050 Laptop GPU",
        tag="test_save",
    )

    assert "# PROJECT NOIR — AUTONOMOUS TRAINING REPORT" in report_text
    assert "Explain Like I'm Five" in report_text
    assert "Quantum Computing Fundamentals" in report_text
    assert "47.20" in report_text
    assert "Curiosity" in report_text
    assert "The future of intelligence is expanding rapidly." in report_text


def test_save_report_files(tmp_path):
    target_dir = tmp_path / "checkpoint_01"
    target_dir.mkdir()

    summary_file = TrainingReportGenerator.save_report_files(
        experiment_id="test_exp_save",
        step=10,
        epoch=1,
        metrics={"train_loss": 4.12, "perplexity": 61.5},
        target_dirs=[target_dir],
    )

    assert summary_file.exists()
    assert (target_dir / "REPORT.md").exists()
    assert (Path("docs") / "LATEST_LEARNING_REPORT.md").exists()
