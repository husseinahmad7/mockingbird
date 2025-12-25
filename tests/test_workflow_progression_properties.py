"""Property-based tests for workflow progression correctness.

**Feature: video-translator, Property 19: Workflow progression correctness**
**Validates: Requirements 9.5**

Property 19: Workflow progression correctness
For any workflow state transitions, the system should maintain valid state
and allow progression only when prerequisites are met.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import List, Dict, Any
from dataclasses import dataclass

from src.models.core import JobStatus


@dataclass
class WorkflowState:
    """Represents the state of the workflow."""
    current_step: str
    uploaded_file_path: str | None
    transcription_complete: bool
    translation_complete: bool
    review_complete: bool
    can_proceed: bool


class WorkflowManager:
    """Manages workflow progression logic."""
    
    VALID_STEPS = ['upload', 'transcribe', 'translate', 'review', 'export']
    
    @staticmethod
    def can_proceed_to_next_step(state: WorkflowState) -> bool:
        """Check if workflow can proceed to next step.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if can proceed, False otherwise
        """
        if state.current_step == 'upload':
            return state.uploaded_file_path is not None
        elif state.current_step == 'transcribe':
            return state.transcription_complete
        elif state.current_step == 'translate':
            return state.translation_complete
        elif state.current_step == 'review':
            return state.review_complete
        elif state.current_step == 'export':
            return False  # Export is final step
        else:
            return False
    
    @staticmethod
    def get_next_step(current_step: str) -> str | None:
        """Get the next step in the workflow.
        
        Args:
            current_step: Current workflow step
            
        Returns:
            Next step name, or None if at end
        """
        try:
            current_idx = WorkflowManager.VALID_STEPS.index(current_step)
            if current_idx < len(WorkflowManager.VALID_STEPS) - 1:
                return WorkflowManager.VALID_STEPS[current_idx + 1]
            return None
        except ValueError:
            return None
    
    @staticmethod
    def get_previous_step(current_step: str) -> str | None:
        """Get the previous step in the workflow.
        
        Args:
            current_step: Current workflow step
            
        Returns:
            Previous step name, or None if at beginning
        """
        try:
            current_idx = WorkflowManager.VALID_STEPS.index(current_step)
            if current_idx > 0:
                return WorkflowManager.VALID_STEPS[current_idx - 1]
            return None
        except ValueError:
            return None
    
    @staticmethod
    def is_valid_step(step: str) -> bool:
        """Check if step is valid.
        
        Args:
            step: Step name to check
            
        Returns:
            True if valid step, False otherwise
        """
        return step in WorkflowManager.VALID_STEPS
    
    @staticmethod
    def get_step_index(step: str) -> int:
        """Get the index of a step in the workflow.
        
        Args:
            step: Step name
            
        Returns:
            Index of step, or -1 if invalid
        """
        try:
            return WorkflowManager.VALID_STEPS.index(step)
        except ValueError:
            return -1


class TestWorkflowProgressionProperties:
    """Property-based tests for workflow progression correctness."""
    
    @given(
        current_step=st.sampled_from(['upload', 'transcribe', 'translate', 'review', 'export'])
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_step_recognition_property(self, current_step):
        """Property: All workflow steps should be recognized as valid.
        
        For any step in the workflow, the system should recognize it as valid.
        """
        assert WorkflowManager.is_valid_step(current_step), \
            f"Step '{current_step}' should be recognized as valid"
    
    @given(
        invalid_step=st.text(min_size=1, max_size=50).filter(
            lambda s: s not in ['upload', 'transcribe', 'translate', 'review', 'export']
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_step_rejection_property(self, invalid_step):
        """Property: Invalid steps should be rejected.
        
        For any string that is not a valid workflow step, the system should
        recognize it as invalid.
        """
        assert not WorkflowManager.is_valid_step(invalid_step), \
            f"Step '{invalid_step}' should be recognized as invalid"
    
    @given(
        current_step=st.sampled_from(['upload', 'transcribe', 'translate', 'review'])
    )
    @settings(max_examples=100, deadline=None)
    def test_next_step_progression_property(self, current_step):
        """Property: Each step (except last) should have a valid next step.
        
        For any non-final step, there should be a valid next step in the workflow.
        """
        next_step = WorkflowManager.get_next_step(current_step)
        
        assert next_step is not None, \
            f"Step '{current_step}' should have a next step"
        
        assert WorkflowManager.is_valid_step(next_step), \
            f"Next step '{next_step}' should be valid"
        
        # Property: Next step should come after current step
        current_idx = WorkflowManager.get_step_index(current_step)
        next_idx = WorkflowManager.get_step_index(next_step)
        
        assert next_idx == current_idx + 1, \
            f"Next step should be immediately after current step"
    
    def test_final_step_has_no_next_property(self):
        """Property: The final step should have no next step.
        
        The export step should be the final step with no next step.
        """
        next_step = WorkflowManager.get_next_step('export')
        
        assert next_step is None, \
            "Export step should have no next step"
    
    @given(
        current_step=st.sampled_from(['transcribe', 'translate', 'review', 'export'])
    )
    @settings(max_examples=100, deadline=None)
    def test_previous_step_navigation_property(self, current_step):
        """Property: Each step (except first) should have a valid previous step.
        
        For any non-initial step, there should be a valid previous step.
        """
        prev_step = WorkflowManager.get_previous_step(current_step)
        
        assert prev_step is not None, \
            f"Step '{current_step}' should have a previous step"
        
        assert WorkflowManager.is_valid_step(prev_step), \
            f"Previous step '{prev_step}' should be valid"
        
        # Property: Previous step should come before current step
        current_idx = WorkflowManager.get_step_index(current_step)
        prev_idx = WorkflowManager.get_step_index(prev_step)
        
        assert prev_idx == current_idx - 1, \
            f"Previous step should be immediately before current step"
    
    def test_first_step_has_no_previous_property(self):
        """Property: The first step should have no previous step.
        
        The upload step should be the first step with no previous step.
        """
        prev_step = WorkflowManager.get_previous_step('upload')
        
        assert prev_step is None, \
            "Upload step should have no previous step"
    
    @given(
        has_file=st.booleans()
    )
    @settings(max_examples=50, deadline=None)
    def test_upload_step_prerequisite_property(self, has_file):
        """Property: Upload step can proceed only if file is uploaded.
        
        For the upload step, progression should be allowed only when a file
        has been uploaded.
        """
        state = WorkflowState(
            current_step='upload',
            uploaded_file_path='/path/to/file.mp4' if has_file else None,
            transcription_complete=False,
            translation_complete=False,
            review_complete=False,
            can_proceed=False
        )
        
        can_proceed = WorkflowManager.can_proceed_to_next_step(state)
        
        assert can_proceed == has_file, \
            f"Upload step should {'allow' if has_file else 'block'} progression"
    
    @given(
        transcription_complete=st.booleans()
    )
    @settings(max_examples=50, deadline=None)
    def test_transcribe_step_prerequisite_property(self, transcription_complete):
        """Property: Transcribe step can proceed only if transcription is complete.
        
        For the transcribe step, progression should be allowed only when
        transcription is complete.
        """
        state = WorkflowState(
            current_step='transcribe',
            uploaded_file_path='/path/to/file.mp4',
            transcription_complete=transcription_complete,
            translation_complete=False,
            review_complete=False,
            can_proceed=False
        )
        
        can_proceed = WorkflowManager.can_proceed_to_next_step(state)
        
        assert can_proceed == transcription_complete, \
            f"Transcribe step should {'allow' if transcription_complete else 'block'} progression"

    @given(
        translation_complete=st.booleans()
    )
    @settings(max_examples=50, deadline=None)
    def test_translate_step_prerequisite_property(self, translation_complete):
        """Property: Translate step can proceed only if translation is complete.

        For the translate step, progression should be allowed only when
        translation is complete.
        """
        state = WorkflowState(
            current_step='translate',
            uploaded_file_path='/path/to/file.mp4',
            transcription_complete=True,
            translation_complete=translation_complete,
            review_complete=False,
            can_proceed=False
        )

        can_proceed = WorkflowManager.can_proceed_to_next_step(state)

        assert can_proceed == translation_complete, \
            f"Translate step should {'allow' if translation_complete else 'block'} progression"

    @given(
        review_complete=st.booleans()
    )
    @settings(max_examples=50, deadline=None)
    def test_review_step_prerequisite_property(self, review_complete):
        """Property: Review step can proceed only if review is complete.

        For the review step, progression should be allowed only when
        review is complete.
        """
        state = WorkflowState(
            current_step='review',
            uploaded_file_path='/path/to/file.mp4',
            transcription_complete=True,
            translation_complete=True,
            review_complete=review_complete,
            can_proceed=False
        )

        can_proceed = WorkflowManager.can_proceed_to_next_step(state)

        assert can_proceed == review_complete, \
            f"Review step should {'allow' if review_complete else 'block'} progression"

    def test_export_step_is_final_property(self):
        """Property: Export step should not allow further progression.

        The export step is the final step and should not allow progression
        to any next step.
        """
        state = WorkflowState(
            current_step='export',
            uploaded_file_path='/path/to/file.mp4',
            transcription_complete=True,
            translation_complete=True,
            review_complete=True,
            can_proceed=False
        )

        can_proceed = WorkflowManager.can_proceed_to_next_step(state)

        assert not can_proceed, \
            "Export step should not allow further progression"

    @given(
        steps=st.lists(
            st.sampled_from(['upload', 'transcribe', 'translate', 'review', 'export']),
            min_size=2,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_step_ordering_consistency_property(self, steps):
        """Property: Step indices should maintain consistent ordering.

        For any set of unique steps, their indices should reflect their
        position in the workflow.
        """
        # Sort steps by their index
        sorted_steps = sorted(steps, key=lambda s: WorkflowManager.get_step_index(s))

        # Property: Indices should be in ascending order
        indices = [WorkflowManager.get_step_index(s) for s in sorted_steps]

        for i in range(len(indices) - 1):
            assert indices[i] < indices[i + 1], \
                "Step indices should be in ascending order"

    @given(
        num_steps=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_forward_navigation_chain_property(self, num_steps):
        """Property: Forward navigation should form a valid chain.

        For any number of forward steps, each next step should be reachable
        and valid.
        """
        current = 'upload'
        visited = [current]

        for _ in range(min(num_steps, len(WorkflowManager.VALID_STEPS) - 1)):
            next_step = WorkflowManager.get_next_step(current)

            if next_step is None:
                break

            # Property: Next step should be valid
            assert WorkflowManager.is_valid_step(next_step), \
                f"Next step '{next_step}' should be valid"

            # Property: Should not revisit steps
            assert next_step not in visited, \
                f"Should not revisit step '{next_step}'"

            visited.append(next_step)
            current = next_step

        # Property: All visited steps should be in correct order
        for i in range(len(visited) - 1):
            idx1 = WorkflowManager.get_step_index(visited[i])
            idx2 = WorkflowManager.get_step_index(visited[i + 1])

            assert idx2 == idx1 + 1, \
                "Forward navigation should visit steps in sequence"

    @given(
        num_steps=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_backward_navigation_chain_property(self, num_steps):
        """Property: Backward navigation should form a valid chain.

        For any number of backward steps, each previous step should be
        reachable and valid.
        """
        current = 'export'
        visited = [current]

        for _ in range(min(num_steps, len(WorkflowManager.VALID_STEPS) - 1)):
            prev_step = WorkflowManager.get_previous_step(current)

            if prev_step is None:
                break

            # Property: Previous step should be valid
            assert WorkflowManager.is_valid_step(prev_step), \
                f"Previous step '{prev_step}' should be valid"

            # Property: Should not revisit steps
            assert prev_step not in visited, \
                f"Should not revisit step '{prev_step}'"

            visited.append(prev_step)
            current = prev_step

        # Property: All visited steps should be in reverse order
        for i in range(len(visited) - 1):
            idx1 = WorkflowManager.get_step_index(visited[i])
            idx2 = WorkflowManager.get_step_index(visited[i + 1])

            assert idx2 == idx1 - 1, \
                "Backward navigation should visit steps in reverse sequence"

