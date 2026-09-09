"""Compatibility exports for the isolated page-section semantic adapter."""

from chronovisor.core.page_evidence import (
    SECTION_SEMANTIC_KIND,
    PageSectionSemanticError,
    build_page_section_documents,
    resolve_page_section_evidence,
)

__all__ = [
    "SECTION_SEMANTIC_KIND",
    "PageSectionSemanticError",
    "build_page_section_documents",
    "resolve_page_section_evidence",
]
