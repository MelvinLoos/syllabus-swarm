"""syllabus-swarm — exporters package."""

from src.exporters.file_writer import (
    FileWriteError,
    OutputPathConfig,
    write_directory_tree,
    write_file,
    write_syllabus,
    write_lab_file,
    write_rubric,
    OUTPUT_PATHS,
)

from src.exporters.manifest import (
    ArtifactSummary,
    ManifestData,
    update_output_manifest,
)

__all__ = [
    # file_writer
    "FileWriteError",
    "OutputPathConfig",
    "OUTPUT_PATHS",
    "write_directory_tree",
    "write_file",
    "write_syllabus",
    "write_lab_file",
    "write_rubric",
    # manifest
    "ArtifactSummary",
    "ManifestData",
    "update_output_manifest",
]