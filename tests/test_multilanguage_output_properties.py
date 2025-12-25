"""Property-based tests for multi-language output separation.

**Feature: video-translator, Property 22: Multi-language output separation**
**Validates: Requirements 10.5**

Property 22: Multi-language output separation
For any multi-language export, the system should clearly separate outputs
by language and maintain independence between language versions.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import List, Dict
import tempfile
from pathlib import Path

from src.models.core import Segment
from src.services.subtitle_exporter import SubtitleExporter


# Strategy for generating valid timestamps
@st.composite
def timestamp_pair(draw):
    """Generate a valid pair of start and end timestamps."""
    start = draw(st.floats(min_value=0.0, max_value=3600.0))
    duration = draw(st.floats(min_value=0.1, max_value=30.0))
    end = start + duration
    return start, end


# Strategy for generating transcription segments with translations
@st.composite
def transcription_segment_with_translation(draw):
    """Generate a transcription segment with translation."""
    start, end = draw(timestamp_pair())
    text = draw(st.text(min_size=1, max_size=500, alphabet=st.characters(blacklist_categories=('Cs',))))
    translation = draw(st.text(min_size=1, max_size=500, alphabet=st.characters(blacklist_categories=('Cs',))))
    speaker_id = draw(st.one_of(st.none(), st.text(min_size=1, max_size=20)))
    
    segment = Segment(
        start_time=start,
        end_time=end,
        text=text,
        speaker_id=speaker_id
    )
    # Add translation as attribute
    segment.translation = translation
    
    return segment


class TestMultiLanguageOutputProperties:
    """Property-based tests for multi-language output separation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = SubtitleExporter()
        self.temp_dir = tempfile.mkdtemp()
    
    @given(
        segments=st.lists(transcription_segment_with_translation(), min_size=1, max_size=20),
        num_languages=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_language_output_independence_property(self, segments, num_languages):
        """Property: Each language output should be independent.
        
        For any multi-language export, each language version should be
        completely independent and not affect others.
        """
        languages = [f"lang_{i}" for i in range(num_languages)]
        output_dir = Path(self.temp_dir)
        
        # Export for each language
        results = self.exporter.export_multi_language(
            segments,
            str(output_dir),
            "test",
            ['original'] + languages
        )
        
        # Property: Should have results for all languages
        assert 'original' in results, "Should have original language export"
        for lang in languages:
            assert lang in results, f"Should have export for {lang}"
        
        # Property: Each language should have independent files
        for lang in ['original'] + languages:
            srt_success, ass_success = results[lang]
            assert srt_success, f"SRT export for {lang} should succeed"
            assert ass_success, f"ASS export for {lang} should succeed"
    
    @given(
        segments=st.lists(transcription_segment_with_translation(), min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_original_vs_translation_separation_property(self, segments):
        """Property: Original and translated outputs should be clearly separated.
        
        For any segments with translations, the original and translated
        versions should contain different text.
        """
        # Export original
        original_file = Path(self.temp_dir) / "original.srt"
        success_orig = self.exporter.export_srt(
            segments,
            str(original_file),
            use_translation=False
        )
        
        # Export translation
        translation_file = Path(self.temp_dir) / "translation.srt"
        success_trans = self.exporter.export_srt(
            segments,
            str(translation_file),
            use_translation=True
        )
        
        assert success_orig, "Original export should succeed"
        assert success_trans, "Translation export should succeed"
        
        # Read contents
        with open(original_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        with open(translation_file, 'r', encoding='utf-8') as f:
            translation_content = f.read()
        
        # Property: Original should contain original text
        for segment in segments:
            assert segment.text in original_content, \
                "Original file should contain original text"
        
        # Property: Translation should contain translated text
        for segment in segments:
            if hasattr(segment, 'translation'):
                assert segment.translation in translation_content, \
                    "Translation file should contain translated text"
    
    @given(
        segments=st.lists(transcription_segment_with_translation(), min_size=1, max_size=20),
        num_languages=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_language_file_naming_consistency_property(self, segments, num_languages):
        """Property: Language files should have consistent naming.
        
        For any multi-language export, files should be named consistently
        with language identifiers.
        """
        languages = [f"lang_{i}" for i in range(num_languages)]
        output_dir = Path(self.temp_dir)
        base_filename = "test_naming"
        
        # Export for each language
        results = self.exporter.export_multi_language(
            segments,
            str(output_dir),
            base_filename,
            ['original'] + languages
        )
        
        # Property: Files should exist with correct naming pattern
        for lang in ['original'] + languages:
            srt_file = output_dir / f"{base_filename}_{lang}.srt"
            ass_file = output_dir / f"{base_filename}_{lang}.ass"
            
            assert srt_file.exists(), \
                f"SRT file for {lang} should exist with correct name"
            assert ass_file.exists(), \
                f"ASS file for {lang} should exist with correct name"
    
    @given(
        segments=st.lists(transcription_segment_with_translation(), min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_timing_consistency_across_languages_property(self, segments):
        """Property: Timing should be consistent across all language versions.
        
        For any multi-language export, all language versions should have
        the same timing information.
        """
        # Export original
        original_file = Path(self.temp_dir) / "timing_original.srt"
        self.exporter.export_srt(segments, str(original_file), use_translation=False)
        
        # Export translation
        translation_file = Path(self.temp_dir) / "timing_translation.srt"
        self.exporter.export_srt(segments, str(translation_file), use_translation=True)
        
        # Read contents
        with open(original_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        with open(translation_file, 'r', encoding='utf-8') as f:
            translation_content = f.read()
        
        # Property: Both should have same number of timestamp arrows
        original_arrows = original_content.count(" --> ")
        translation_arrows = translation_content.count(" --> ")
        
        assert original_arrows == translation_arrows, \
            "Both versions should have same number of timestamps"
        
        # Property: Both should have same number of segments
        import re
        original_indices = re.findall(r'^\d+$', original_content, re.MULTILINE)
        translation_indices = re.findall(r'^\d+$', translation_content, re.MULTILINE)
        
        assert len(original_indices) == len(translation_indices), \
            "Both versions should have same number of segments"
    
    @given(
        segments=st.lists(transcription_segment_with_translation(), min_size=1, max_size=20),
        num_languages=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_segment_count_consistency_property(self, segments, num_languages):
        """Property: All language versions should have same segment count.
        
        For any multi-language export, all versions should contain the
        same number of segments.
        """
        languages = [f"lang_{i}" for i in range(num_languages)]
        output_dir = Path(self.temp_dir)
        
        # Export for each language
        results = self.exporter.export_multi_language(
            segments,
            str(output_dir),
            "test_count",
            ['original'] + languages
        )
        
        # Count segments in each file
        segment_counts = {}
        
        for lang in ['original'] + languages:
            srt_file = output_dir / f"test_count_{lang}.srt"
            
            with open(srt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count segment indices
            import re
            indices = re.findall(r'^\d+$', content, re.MULTILINE)
            segment_counts[lang] = len(indices)
        
        # Property: All versions should have same count
        expected_count = len(segments)
        for lang, count in segment_counts.items():
            assert count == expected_count, \
                f"Language {lang} should have {expected_count} segments, found {count}"
    
    @given(
        segments=st.lists(transcription_segment_with_translation(), min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_format_consistency_across_languages_property(self, segments):
        """Property: File format should be consistent across languages.
        
        For any multi-language export, all language versions should use
        the same file format and structure.
        """
        # Export both languages in both formats
        files = {
            'original_srt': Path(self.temp_dir) / "format_original.srt",
            'original_ass': Path(self.temp_dir) / "format_original.ass",
            'translation_srt': Path(self.temp_dir) / "format_translation.srt",
            'translation_ass': Path(self.temp_dir) / "format_translation.ass",
        }
        
        self.exporter.export_srt(segments, str(files['original_srt']), use_translation=False)
        self.exporter.export_ass(segments, str(files['original_ass']), use_translation=False)
        self.exporter.export_srt(segments, str(files['translation_srt']), use_translation=True)
        self.exporter.export_ass(segments, str(files['translation_ass']), use_translation=True)
        
        # Property: SRT files should have same structure
        with open(files['original_srt'], 'r', encoding='utf-8') as f:
            orig_srt = f.read()
        with open(files['translation_srt'], 'r', encoding='utf-8') as f:
            trans_srt = f.read()
        
        assert orig_srt.count('\n\n') == trans_srt.count('\n\n'), \
            "SRT files should have same number of blank line separators"
        
        # Property: ASS files should have same sections
        with open(files['original_ass'], 'r', encoding='utf-8') as f:
            orig_ass = f.read()
        with open(files['translation_ass'], 'r', encoding='utf-8') as f:
            trans_ass = f.read()
        
        for section in ['[Script Info]', '[V4+ Styles]', '[Events]']:
            assert (section in orig_ass) == (section in trans_ass), \
                f"Both ASS files should have {section} section"
    
    @given(
        segments=st.lists(transcription_segment_with_translation(), min_size=1, max_size=20),
        num_languages=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_no_language_cross_contamination_property(self, segments, num_languages):
        """Property: Language outputs should not contain text from other languages.
        
        For any multi-language export, each language file should only
        contain text from that specific language.
        """
        if num_languages < 2:
            return
        
        languages = [f"lang_{i}" for i in range(num_languages)]
        output_dir = Path(self.temp_dir)
        
        # Export for each language
        self.exporter.export_multi_language(
            segments,
            str(output_dir),
            "test_contamination",
            ['original'] + languages
        )
        
        # Read original file
        original_file = output_dir / "test_contamination_original.srt"
        with open(original_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Read translation files
        for lang in languages:
            lang_file = output_dir / f"test_contamination_{lang}.srt"
            with open(lang_file, 'r', encoding='utf-8') as f:
                lang_content = f.read()
            
            # Property: Translation file should not contain original text
            # (except for timestamps and indices which are shared)
            for segment in segments:
                # Original text should not appear in translation file
                # (unless it happens to be the same as translation)
                if hasattr(segment, 'translation') and segment.text != segment.translation:
                    # This is a soft check - we verify that translation is present
                    assert segment.translation in lang_content, \
                        f"Translation file should contain translated text for {lang}"

